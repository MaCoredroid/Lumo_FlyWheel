# Round F_b Speed Design: Adaptive Batched Paths With Row-Scaled K

**Generated:** 2026-05-26
**Scope:** Qwen3.6/Qwen3-Next-style native MTP on vLLM 0.19.x, Track-B serving
stack, batch-1 agent decode.
**Status:** Design proposal. Do not treat as shipped behavior.
**Primary objective:** make F_b lossless and fast by ensuring K path growth is
paid mostly as wider batched rows in existing model/GEMM/kernel work, not as K
extra sequential proposer/verifier/control-flow passes.

---

## 1. Executive Summary

F_b should be the shipping path for multi-candidate MTP on Qwen3.6-style hybrid
Gated DeltaNet models. F_a packed-tree verification is blocked because
TreeAttention masks softmax attention but does not fork GDN/Mamba recurrent
state. F_b avoids that by verifying each candidate as an independent linear
history.

The first recovered F_b implementation proves the right semantic shape but is
too expensive:

- proposer path extension is sequential for K=2;
- scheduler clones are materialized as temporary requests;
- GDN/Mamba state blocks are copied eagerly;
- K=2 is used even when the second path has low expected value.

This design replaces fixed-K F_b with **adaptive row-scaled F_b**:

1. Generate candidate roots from first-step MTP logits.
2. Decide K dynamically from root distribution and recent acceptance history.
3. Extend K candidate paths in one batched drafter loop where K appears as an
   extra batch row dimension.
4. Verify K paths in one target forward as K verifier rows.
5. Run lossless rejection per row with shared logical-root sampling where
   required.
6. Collapse only the winning row back into the parent request.
7. Optimize state management toward copy-on-write so K mostly adds rows, not
   prefix copies.

The ideal cost target is:

```text
K=1:  one row through drafter + one row through target verifier
K=2:  two rows through the same drafter/target kernels, not two serial calls
K=N:  N rows in batch dimension, bounded by occupancy/cache pressure
```

K must never mean "repeat the whole speculative decode pipeline K times."

---

## 2. Background And Constraints

### 2.1 Hybrid model constraint

Qwen3-Next/Qwen3.6-style models are not vanilla transformer-only models. Public
Qwen/vLLM docs describe a hybrid stack with Gated DeltaNet linear attention and
full attention, plus a hybrid KV/cache manager. That makes branch verification
a state-forking problem, not only an attention-mask problem.

Relevant public references:

- vLLM Qwen3-Next support blog:
  <https://vllm.ai/blog/2025-09-11-qwen3-next>
- Qwen3-Next FP8 model card:
  <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8>
- vLLM P-EAGLE blog, useful precedent for parallelizing draft generation:
  <https://vllm.ai/blog/2026-03-13-p-eagle>

### 2.2 Existing repo evidence

Current recovered F_b lives in:

- `scripts/swe_x86_helpers/relaunch_qwen36_round.py`
  - `_FB_BLOCK`
  - `LUMO_FB_PATHS_EAGLE`
  - `LUMO_FB_PATHS_SCHED`
  - `LUMO_FB_PATHS_RUNNER`
  - `LUMO_FB_SHARED_ROOT_SAMPLE`

Current top-level runner gap:

- `scripts/run_codex_experiment.py` does not expose `Fb` in `--config`
  choices or `apply_config`.

Prior F_a/F_b reports:

- `docs/reports/auto_research/round-F-tree-rejection-sampler-KT-20260526.md`
- `docs/reports/auto_research/round-Fb-batched-paths-blocker-audit-20260526.md`

### 2.3 Non-negotiable correctness contract

F_b is allowed to change scheduling and proposal order. It is not allowed to
change the target distribution.

Lossless contract:

- each candidate path row must start from the exact parent target state;
- sibling rows must not mutate shared GDN/Mamba recurrent state;
- rejection sampling must be equivalent to the ordinary flat-chain target rule;
- winner collapse must leave parent tokens, full-attention KV, GDN/Mamba state,
  sampling state, and counters equal to the state that would exist if the winner
  path had been verified as the only path.

---

## 3. Design Goal: Row-Scaled K

The central invariant is the **row-scaled K invariant**:

> Increasing K should add rows to existing tensor batches. It should not add K
> independent Python scheduler passes, K independent drafter loops, or K prefix
> state copies.

### 3.1 Desired tensor shape

Use K as a batch-row dimension:

```text
parent batch B = 1
candidate paths K = 1 or 2 initially
spec depth D = 3
hidden H
vocab V

root logits:        [B, V]
root candidates:    [B, K]
path token ids:     [B, K, D]
path hidden state:  [B*K, H]
target verify rows: [B*K, D+1]
target logits:      [B*K, D+1, V] or flattened [sum rows, V]
acceptance:         [B, K]
winner:             [B]
```

The current recovered implementation returns flattened path ids like:

```text
[path0_tok0, path0_tok1, path0_tok2, path1_tok0, path1_tok1, path1_tok2]
```

That is acceptable as a transport format only. Internally, the proposer and
verifier should treat it as `[K, D]`.

### 3.2 Cost target

Let:

- `C_target(B)` be target forward cost for B rows;
- `C_draft(B)` be MTP drafter forward cost for B rows;
- `C_sched(K)` be Python/control overhead;
- `C_state(K)` be state clone/collapse overhead.

Current fixed K=2 design tends toward:

```text
C ~= 2*C_draft(1) + C_target(2) + C_sched(2 requests) + C_state(eager copies)
```

Target design:

```text
C ~= C_draft(K) + C_target(K) + C_sched(1 logical request) + C_state(COW delta)
```

For small K on GPU, `C_draft(2)` and `C_target(2)` should be much closer to one
larger batched call than to two separate calls. This is the only way F_b can
beat E3 consistently.

---

## 4. Architecture

### 4.1 Components

F_b should be split into five explicit components:

| Component | Responsibility | Current recovered surface | Desired surface |
|---|---|---|---|
| Root selector | choose top candidate roots and decide K | `_lumo_fb_propose` inline | `FbRootPolicy` |
| Batched path proposer | extend K roots to depth D | sequential `_lumo_fb_extend_one` calls | `extend_paths_batched(K,D)` |
| Path-row verifier | schedule/verify K linear rows | temporary request clones | internal path rows |
| Rejection/collapse | accept per row, choose winner | scheduler update wrapper | explicit `FbCollapseResult` |
| State manager | fork/copy/commit KV and GDN/Mamba state | eager block clone/copy | path state COW |

The launcher patch can continue to prototype, but the durable implementation
should move toward a vLLM fork or repo-owned patch module. The current heredoc
has crossed the line where local reasoning is hard.

### 4.2 Data model

Introduce conceptual records, even if first implemented as Python dicts:

```python
class FbPathBatch:
    parent_req_id: str
    k: int
    depth: int
    root_token_ids: Tensor  # [K]
    path_token_ids: Tensor  # [K, D]
    root_scores: Tensor     # [K]
    policy_reason: str

class FbPathRow:
    parent_req_id: str
    row_id: str
    path_idx: int
    token_ids: list[int]    # length D
    state_ref: PathStateRef

class FbCollapseResult:
    parent_req_id: str
    winner_row_id: str
    loser_row_ids: list[str]
    accepted_len: int
    committed_token_ids: list[int]
    state_commit: StateCommitPlan
```

This separation matters. Proposal tensors, verifier rows, and committed request
state have different lifetimes.

---

## 5. Adaptive K Policy

### 5.1 Why adaptive K is mandatory

Fixed K=2 is wasteful in common cases:

- if root top-1 is very confident, path1 is unlikely to win;
- if both candidates would reject at root, the second verifier row is pure loss;
- if path0 accepts D tokens, path1 can only tie or beat through sampling edge
  cases and may not justify its cost;
- if recent accepted length is high, E3/K=1 is already doing well.

F_b should spend K=2 only on uncertain root distributions.

### 5.2 Inputs

Policy inputs:

- first-step MTP root logits or probabilities;
- `p1`, `p2`, `margin = logp1 - logp2`;
- entropy over top N roots;
- recent acceptance moving average for the request;
- recent F_b K=2 win rate;
- current queue pressure/GPU occupancy, if available;
- temperature and sampling mode.

### 5.3 Initial rule

Start with a simple deterministic policy:

```text
Use K=1 if:
  p1 >= 0.70
  or p2 / p1 <= 0.20
  or recent_acc_len >= 2.4

Use K=2 if:
  p1 < 0.70
  and p2 / p1 > 0.20
  and recent K=2 overhead budget is not exceeded
```

The thresholds are deliberately conservative. They should be tuned from
`spec_speed_probe` traces, not guessed permanently.

### 5.4 Bandit upgrade

After the static policy works, replace thresholds with a cheap contextual
bandit:

```text
reward = committed_tokens - alpha * extra_rows - beta * extra_state_copy_us
arms = {K=1, K=2}
features = [p1, p2/p1, entropy, recent_acc, temp, prompt_position_bucket]
```

Do not train or persist a complex model initially. A per-run online policy with
bounded exploration is enough to discover when K=2 is profitable.

---

## 6. Batched Path Proposer

### 6.1 Current problem

Current F_b effectively does:

```python
path0 = extend_one(root0)
path1 = extend_one(root1)
out = cat(path0, path1)
```

That makes K=2 pay two drafter extension loops. It violates the row-scaled K
invariant.

### 6.2 Desired algorithm

Use a batched proposer:

```python
def extend_paths_batched(roots: Tensor[K], base_state, depth: int) -> Tensor[K, D]:
    path_ids = zeros([K, D])
    path_ids[:, 0] = roots

    hidden = repeat(base_hidden_state, K)       # [K, H]
    positions = repeat(base_position, K)        # [K] or M-RoPE form
    attn_meta = build_draft_metadata(rows=K)

    for t in range(1, D):
        input_ids = path_ids[:, t - 1]          # [K]
        hidden = mtp_draft_forward(
            input_ids=input_ids,
            hidden_states=hidden,
            positions=positions,
            metadata=attn_meta,
        )                                      # one batched call
        path_ids[:, t] = greedy_or_policy_sample(hidden)
        positions += 1
        attn_meta.advance_one_step()

    return path_ids
```

Key implementation details:

- K rows must have independent draft cache slots for generated suffix tokens.
- Prefix state may be shared read-only.
- Positions/M-RoPE must be row-repeated, not flattened tree positions.
- The code path must not instantiate `TreeAttentionMetadata`.
- For K=1, this should reduce to the standard E3 path as closely as possible.

### 6.3 Expected speed impact

This is likely the largest speed lever. It changes proposer cost from:

```text
2 serial draft forwards per depth step
```

to:

```text
1 draft forward with batch rows=2 per depth step
```

That is the closest practical version of "one more row on matrix multiplication
as K grows."

---

## 7. Path-Row Verification

### 7.1 Current clone approach

The recovered F_b scheduler creates temporary request clones:

```text
parent -> parent::lumo_fb::0
parent -> parent::lumo_fb::1
```

This is semantically understandable but costly:

- Python request creation;
- scheduler list mutation;
- block manager metadata mutation;
- GPU runner request ID replacement;
- loser cleanup;
- debug JSON writes when enabled.

### 7.2 Desired internal row approach

Longer term, do not model path rows as public requests. Model them as internal
verifier rows attached to one parent request:

```text
parent request
  path_row[0]
  path_row[1]
```

The scheduler output should carry:

```python
scheduled_fb_path_rows = {
    parent_req_id: [
        FbPathRow(path_idx=0, token_ids=[...], state_ref=...),
        FbPathRow(path_idx=1, token_ids=[...], state_ref=...),
    ]
}
```

The GPU runner should expand these into B*K rows for target forward, but the
engine should still have one logical request.

### 7.3 Staged path

Stage 1 can keep temporary request clones while correctness stabilizes.

Stage 2 should remove public clone request churn and move to internal path rows.
This is where most Python overhead should disappear.

---

## 8. State Forking And Copy-On-Write

### 8.1 Correctness need

Every path row must observe the same parent prefix state and then mutate an
isolated suffix state. This applies to:

- full-attention KV blocks;
- GDN/Mamba recurrent state blocks;
- M-RoPE/position counters;
- accepted-token counters used by `preprocess_mamba`;
- request-local sampling state where relevant.

### 8.2 Current eager-copy behavior

Current F_b copies Mamba/GDN blocks when clones are created. This is simple but
can dominate runtime if it copies large state every speculative event.

### 8.3 Desired COW model

Use a path state reference:

```text
PathStateRef
  prefix_blocks: shared read-only
  mutable_suffix_blocks: path-owned
  recurrent_state_ref: shared until first write
  recurrent_state_delta: path-owned after fork
```

At fork time:

```text
no full prefix copy
allocate only per-path mutable suffix slots
copy only the minimal recurrent state needed for the next D target tokens
```

At commit time:

```text
parent.state = winner.state
free loser deltas
retain shared prefix
```

### 8.4 Practical first optimization

Before building full COW, measure and minimize the current copy surface:

- count copied full-attention blocks per event;
- count copied Mamba/GDN blocks per event;
- measure copy time with CUDA events or scoped timers;
- check whether only the last state block is mutable for D=3;
- if yes, copy only the last state block plus partial KV block.

The initial implementation should export:

```json
{
  "fb_state_copy_us": 0,
  "fb_kv_blocks_copied": 0,
  "fb_mamba_blocks_copied": 0,
  "fb_loser_blocks_freed": 0
}
```

---

## 9. Rejection Sampling And Winner Collapse

### 9.1 Per-row rejection

Each path row runs normal flat-chain rejection for its own `[D]` draft. For
greedy decoding, this is straightforward. For sampling, root coupling matters.

### 9.2 Shared logical root

The current recovered code includes `LUMO_FB_SHARED_ROOT_SAMPLE`: K sibling rows
represent alternative proposals for one logical next target token. The target
root sample should be drawn once for the logical request, then compared against
each row's root candidate. Otherwise rows can independently sample different
root target tokens and the K-way comparison is not the same logical event.

Keep this concept, but move it out of a monkey-patched rejection sampler and
into the explicit F_b collapse path.

### 9.3 Winner rule

Initial deterministic winner:

```text
winner = argmax(accepted_len, tie_break=-path_idx)
```

For sampling, preserve the lossless target distribution:

- do not pick a lower-probability path just because it has a better draft score;
- the selected committed token sequence must match the rejection-sampling
  outcome for the shared target root event;
- any recovered token after rejection must be sampled from the correct residual
  distribution.

If this is difficult to prove for non-greedy mode, gate F_b K=2 to greedy or
temperature-controlled canaries first, then expand.

---

## 10. Telemetry And Cost Attribution

F_b needs event-level accounting or speed work will be blind.

Add per-event metrics:

```json
{
  "fb_enabled": true,
  "fb_policy_k": 2,
  "fb_policy_reason": "p2_over_p1",
  "fb_root_p1": 0.42,
  "fb_root_p2": 0.31,
  "fb_proposer_us": 0,
  "fb_verify_us": 0,
  "fb_state_fork_us": 0,
  "fb_state_collapse_us": 0,
  "fb_scheduler_us": 0,
  "fb_rows_verified": 2,
  "fb_winner_idx": 1,
  "fb_accept_lens": [1, 3],
  "fb_committed_len": 3
}
```

Aggregate metrics:

| Metric | Why it matters |
|---|---|
| `decode_tps` | final objective |
| `acc/event` | validates speculative value |
| `P(K=2)` | confirms adaptive policy is selective |
| `K=2 win rate` | second path usefulness |
| `extra rows per committed token` | cost efficiency |
| `fb_proposer_us` | catches sequential proposer regression |
| `fb_state_copy_us` | catches COW need |
| `fb_scheduler_us` | catches clone-request overhead |
| `fb_duplicate_path_delta` | correctness canary |

---

## 11. Evaluation Gates

### Gate 0: runner wiring

- `scripts/run_codex_experiment.py --config Fb --apply-config` works.
- Direct relaunch still works:

```bash
.venv/bin/python scripts/swe_x86_helpers/relaunch_qwen36_round.py --config Fb --mtp 3
```

### Gate 1: K=1 equivalence

Run F_b with `LUMO_FB_K=1`.

Pass criteria:

- no correctness regression versus E3;
- acceptance distribution close to E3;
- overhead measured and attributed.

If K=1 is slower than E3 by more than noise, do not tune K=2 yet.

### Gate 2: duplicate-path K=2

Run with path1 identical to path0.

Pass criteria:

- path0/path1 accepted lengths match, or differences are fully explained by
  shared-root sampling semantics;
- winner collapse leaves parent state valid for subsequent decode;
- no drop in path0 acceptance versus K=1 beyond measurement noise.

This catches sibling state corruption.

### Gate 3: adaptive K=2 correctness

Run real K=2 with adaptive policy enabled.

Pass criteria:

- OFF/E3 byte-exact or approved distributional gate for fixed seed prompts;
- no state divergence after winner collapse;
- K=2 events logged with winner and accept lengths.

### Gate 4: speed

Use `scripts/spec_speed_probe.py` first, then real SWE/agent tasks.

Pass criteria:

- `decode_tps > E3` on the same prompt set;
- `acc/event >= E3` or lower `acc/event` with higher final `decode_tps` must be
  explained by lower overhead;
- `extra rows per committed token` improves over fixed K=2;
- no throughput collapse under longer contexts.

---

## 12. Implementation Plan

### Phase A: make F_b runnable and measured

1. Add `Fb` to `scripts/run_codex_experiment.py` choices.
2. Route `Fb` through `apply_config`.
3. Add labels for `K=1`, `K=2`, duplicate path, and adaptive mode.
4. Add event-level timing around proposer, scheduler clone, state copy, target
   verify, rejection, and collapse.

Deliverable:

- one clean F_b probe matrix comparing `E3`, `Fb-K1`, `Fb-K2-dup`,
  `Fb-K2-fixed`.

### Phase B: adaptive policy

1. Add root probability extraction in `_lumo_fb_propose`.
2. Implement static threshold policy.
3. Return one path when policy chooses K=1 and two paths when it chooses K=2.
4. Log policy features and decision.

Deliverable:

- adaptive F_b probe with `P(K=2)`, K=2 win rate, and decode TPS.

### Phase C: batched proposer

1. Replace sequential `_lumo_fb_extend_one` calls with `extend_paths_batched`.
2. Build draft metadata for K rows.
3. Repeat base hidden state and positions across K rows.
4. Run one MTP forward per depth step for all K rows.
5. Keep K=1 path as a shared code path, not a separate implementation.

Deliverable:

- proposer timing shows K=2 cost closer to batched K rows than 2x K=1.

### Phase D: internal path rows

1. Replace public temporary clone requests with internal verifier rows.
2. Keep one parent request in scheduler-visible state.
3. Teach GPU runner to expand parent into path rows for the target forward.
4. Collapse winner before outputs are emitted.

Deliverable:

- scheduler overhead drops;
- request lifecycle/debugging becomes simpler;
- no clone request IDs leak into external metrics.

### Phase E: state COW

1. Identify exact mutable full-attention and GDN/Mamba state blocks for D=3.
2. Share prefix blocks read-only.
3. Allocate only suffix delta blocks per path.
4. Commit winner deltas by pointer transfer, not copy.
5. Free loser deltas.

Deliverable:

- state-copy metrics fall substantially;
- K=2 memory pressure is bounded.

---

## 13. Risks And Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| K=2 verifier rows still corrupt GDN state | no-ship | duplicate-path gate before speed tuning |
| batched proposer metadata mismatches M-RoPE | bad drafts, low acceptance | K=1 equivalence and top-1 chain invariant |
| shared-root sampling is wrong for temperature mode | distribution shift | start greedy/fixed-seed, add distributional tests before broad sampling |
| internal rows fight vLLM scheduler assumptions | implementation complexity | stage after clone-based correctness and metrics |
| COW state aliasing causes sibling overwrite | silent corruption | state block ID telemetry and duplicate-path canary |
| adaptive K rarely chooses K=2 | no speed gain | tune threshold from trace; add bandit only after static policy works |
| adaptive K chooses K=2 too often | speed loss | budget cap on extra rows per committed token |

---

## 14. Decision Guidance

Prioritize work in this order:

1. **K=1 parity and instrumentation.** Without this, speed data is meaningless.
2. **Batched proposer.** This directly enforces the row-scaled K principle.
3. **Adaptive K.** This avoids paying K=2 when it cannot win.
4. **Internal path rows.** This removes Python request-clone overhead.
5. **State COW.** This reduces recurrent-state copy cost after semantics are
   proven.

Do not spend more time on packed-tree F_a for shipping. F_a needs a branch-aware
GDN kernel, which is a separate research project. F_b can use existing linear
GDN kernels if each path row is a true isolated sequence.

---

## 15. Stop Conditions

Stop or redesign if any of these are true:

- duplicate-path K=2 cannot match K=1 acceptance after state fixes;
- K=2 batched proposer plus internal rows still cannot beat E3 on decode TPS;
- state COW requires invasive vLLM kernel changes comparable to F_a tree-state
  work;
- adaptive K selects K=2 so rarely that complexity is not justified.

If F_b fails these gates, the remaining speed direction should move away from
multi-path verification and toward shallow native MTP tuning, FP8 KV, or model
/ backend selection where speculative decoding is already implemented for the
hybrid state contract.
