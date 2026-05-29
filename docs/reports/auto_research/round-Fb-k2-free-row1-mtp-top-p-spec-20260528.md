# Round F_b K2 Spec: Proposer-Free Row1 From MTP Top-p

**Generated:** 2026-05-28
**Scope:** Qwen3.6/Qwen3-Next-style native MTP on vLLM Track-B, batch-1 agent
decode, F_b internal-row verifier.
**Status:** Near-term implementation spec. Not shipped behavior.
**Primary objective:** make K=2 a strict low-cost extension of K=1 by deriving
row1 from MTP logits already computed while drafting row0, verifying at most one
extra row, and enabling that row only when the expected accept gain can pay for
the verifier cost.

---

## 1. Executive Summary

Current F_b K=2 regresses because it is not only "one more candidate." The
position-tree path computes a canonical K1 trunk, computes additional tree rows,
splices row0 back to the K1 trunk, and verifies extra row material. That keeps
correctness but makes K=2 too expensive for the small acceptance gain we have
measured.

This spec defines the cheaper near-term path:

1. Keep row0 as the exact K1 MTP chain.
2. While drafting row0, retain the unselected top-p/top2 MTP alternatives at
   each draft position.
3. Build row1 from those retained alternatives without any additional
   `_extend_one`, position-tree expansion, or second proposer pass.
4. Verify only one extra row, and only when a confidence gate predicts row0
   likely accepts too few tokens.
5. Log enough data to prove row1 was proposer-free and to decide whether the
   row1 source/gate should ship.
6. Treat unique-node tree packing as a separate later verifier rewrite for K>2.

The guiding invariant is:

```text
K1 cost = row0 proposal + row0 verify
K2 cost = same row0 proposal + one row1 assembled from cached MTP logits + one
          extra verify row only on gated events
```

K2 must not mean a second serial proposer chain or a flattened 2->4->8
position tree.

---

## 2. Why This Is The Right Near-Term Cut

### 2.1 Public systems precedent

TensorRT-LLM's current speculative-decoding docs expose MTP and say MTP can be
combined with Suffix Automaton for higher acceptance on repetitive content:

- <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md>

The older TensorRT-LLM Medusa notes describe the broader tree-verification
principle: candidate paths with common prefixes should be consolidated and
verified through a sparse/tree mask instead of duplicated as independent paths:

- <https://nvidia.github.io/TensorRT-LLM/legacy/advanced/speculative-decoding.html>

SpecInfer, Medusa, and EAGLE-2 point in the same direction:

- <https://arxiv.org/abs/2305.09781>
- <https://arxiv.org/abs/2401.10774>
- <https://arxiv.org/abs/2406.16858>

The useful lesson for us is narrower than "build the whole tree verifier now":
branch candidates should be cheap, context-adaptive, and bounded by verifier
cost. For this Qwen hybrid stack, F_a-style packed trees are a larger project
because recurrent state has to branch with the attention tree. F_b can still
take the cheap candidate-source lesson immediately.

### 2.2 Local implementation evidence

The current code already has a row-scaled path helper:

- `scripts/swe_x86_helpers/relaunch_qwen36_round.py`
  - `_lumo_fb_extend_paths_batched`: advances K linear roots as K draft rows in
    one MTP forward per depth step.

But the current K2 position-tree path does more than that:

- computes the exact K1 trunk with `_lumo_fb_extend_one`;
- computes `_lumo_fb_extend_top2_pos01_tree_batched`;
- splices row0 back to the K1 trunk;
- prunes generated tree rows into single-flip rows;
- verifies the selected flattened rows.

That path is useful for correctness experiments, but it is the wrong cost shape
for a throughput win.

---

## 3. Proposed Algorithm

### 3.1 Row0: exact K1 chain

Row0 must be byte-for-byte the same draft sequence the K1 path would produce
under the same runtime configuration.

Implementation rule:

- reuse the K1 `_lumo_fb_extend_one` behavior as the source of truth initially;
- later, only replace it with a batched implementation after proving row0 is
  batch-invariant against K1.

### 3.2 Row1: free alternative from cached MTP logits

During K1 drafting, each MTP step already produces logits for the next draft
token. Today we greedily choose row0 token and discard the rest. Instead, record
for each draft position:

```text
position i
row0 token: argmax(logits_i)
alt token:  best token in top-p/top2 not equal to row0 token
confidence: p(row0), p(alt), gap, entropy/top-p mass
hidden/state snapshot needed to continue or splice row1
```

Row1 should be a single-flip path:

```text
row0: t0 t1 t2 t3 t4
row1: t0 t1 alt2 u3 u4
```

The flip position is selected by expected value, not fixed topology. Start with
one of these cheap policies:

| Policy | Row1 source | Extra proposer compute | Notes |
|---|---:|---:|---|
| Root alt | `alt0` from first MTP logits | zero | simplest, but prior evidence says root top2 often helps little |
| First low-confidence position | `alti` at earliest low-confidence row0 step | zero if tail is reused conservatively; otherwise bounded tail handling needed | best near-term candidate |
| Best expected-gain position | `argmax p(alt_i) / p(row0_i)` under a low-confidence cap | zero for selection | needs logging first |

Important: if row1 requires a new neural tail after the flip, it is no longer
fully proposer-free. The first implementation should support a strict
`FREE_ROW1_ONLY` mode where row1 is assembled only from already-computed tokens
and cached alternatives, with no extra `_extend_one`.

### 3.3 Verify only one extra row

The near-term K2 verifier shape is exactly two linear rows:

```text
row0 = exact K1 path
row1 = one cached-logit alternative path
```

Disable:

- position-tree expansion;
- 2->4->8 row growth;
- K1 trunk plus separate tree computation;
- row pruning after generating rows that will not be verified.

### 3.4 Adaptive gate

K2 is only useful when row0 likely accepts few tokens. The gate should be based
on values available during row0 drafting:

```text
enable_row1 if:
  predicted_row0_accept_len <= threshold
  and alt_confidence >= min_alt_conf
  and row0_confidence_gap <= max_gap
  and recent_K2_gain_ema > recent_K2_cost_ema
```

Initial static gate for diagnosis:

```text
enable_row1 if min(row0_p_i over i in draft positions) < 0.45
            and best_alt_p_i / row0_p_i > 0.50
```

This mirrors the existing adaptive-root idea but applies it across draft
positions, not only at position 0.

---

## 4. Data Contract And Telemetry

Every speculative event with F_b debug enabled should log one JSON record with:

```json
{
  "event": "fb_free_row1_decision",
  "active_depth": 5,
  "row0": [101, 102, 103, 104, 105],
  "row1": [101, 102, 999, 104, 105],
  "row1_enabled": true,
  "row1_source": "mtp_cached_alt",
  "flip_pos": 2,
  "proposer_free": true,
  "extra_extend_one_calls": 0,
  "position_tree_enabled": false,
  "generated_rows": 2,
  "verified_rows": 2,
  "row0_p": [0.71, 0.66, 0.38, 0.81, 0.74],
  "row1_alt_p": [0.12, 0.14, 0.31, 0.05, 0.08],
  "row0_alt_ratio": [0.17, 0.21, 0.82, 0.06, 0.11],
  "gate_reason": "low_conf_pos2",
  "fb_proposer_us": 12345
}
```

The output probe summary should aggregate:

- gated event count;
- row1 enabled rate;
- row1 source distribution;
- generated rows versus verified rows;
- proposer-free rate;
- extra proposer call count;
- row0 accept length;
- row1 accept length;
- best accept length;
- row1 win rate;
- accepted-token gain per enabled K2 event;
- wall/event cost for K1, gated K2, always-on K2.

Ship blocker: if `proposer_free=false`, `extra_extend_one_calls>0`, or
`generated_rows>verified_rows` for this mode, the run is not measuring this
spec.

---

## 5. Cheap Verification Matrix

Run these before any SWE-Bench-scale claim.

| Run | Purpose | Required knobs |
|---|---|---|
| K1 baseline | establishes row0 throughput and acceptance | `LUMO_FB_K=1`, depth 5 |
| K2 duplicate row | isolates verifier-row overhead | `LUMO_FB_K=2`, `LUMO_FB_DUP_PATH1=1`, no position tree |
| K2 current position tree | regression control | current K2 position-tree mode |
| K2 free row1 shadow | proves candidate value without changing commits | commit row0 only; verify/log row1 |
| K2 free row1 active gated | measures throughput and accepted-token gain | row1 enabled only by gate |
| K2 free row1 always-on | upper-bound acceptance, lower-bound throughput | row1 every event |

Required comparison:

```text
tokens/sec
accepted tokens/event
event_ms
proposer_us
verified rows/event
row1 win rate
row1 accepted-token gain when row0_accept <= 2
```

Expected decision rule:

```text
ship gated K2 only if:
  active_gated_tps > K1_tps
  and active_gated_acc_ev >= K1_acc_ev
  and row0 path remains exact-K1 equivalent
```

If shadow mode shows row1 rarely improves events where row0 accepts <=2, stop.
The issue is candidate quality, not verifier engineering.

---

## 6. Implementation Plan

### Phase 1: instrumentation-only shadow

1. Add a free-row1 mode flag:

```text
LUMO_FB_FREE_ROW1=1
LUMO_FB_FREE_ROW1_SHADOW=1
LUMO_FB_POSITION_TREE=0
```

2. In the row0 `_extend_one` loop, collect per-position top2/top-p candidate
   metadata from already-computed MTP logits.
3. Construct a candidate row1 in Python/Torch without calling `_extend_one`.
4. Verify row1 in shadow if low-risk; otherwise log only the candidate first.
5. Commit row0 only.

### Phase 2: active gated K2

1. Enable row1 verification only when the gate fires.
2. Keep row0 as exact K1 chain.
3. Use exactly two verifier rows when enabled.
4. Collapse winner through existing F_b internal-row path.
5. Preserve no-copy/split-KV correctness checks.

### Phase 3: remove K1 trunk duplicate cost

After row0 batch-invariance is proven, replace the K1 trunk splice with a single
batch-invariant proposer path. Until then, correctness and measurement clarity
matter more than avoiding the row0 single-chain call.

### Phase 4: later K>2 verifier rewrite

Do not implement unique-node packing in this phase. Track it as the long-term
answer for K>2:

```text
candidate trie nodes:
  node_id, parent_id, token_id, depth, source, score

attention:
  each node attends to prefix + ancestor nodes only

hybrid recurrent state:
  read recurrent_state[parent_id]
  write recurrent_state[node_id]

commit:
  adopt only accepted path nodes into parent request
```

This is essentially F_a's packed-tree idea, but Qwen hybrid attention/GDN makes
it a state-tree problem, not just an attention-mask problem. It should be a
separate design/implementation after the cheap K2 source is understood.

---

## 7. Risks And Stop Conditions

| Risk | Symptom | Stop/mitigation |
|---|---|---|
| Row1 is not really free | extra `_extend_one` or tree expansion appears in logs | stop; fix instrumentation before measuring |
| Candidate quality too low | row1 win rate low even when row0 accept <=2 | switch source: suffix/ngram or learned late-branch |
| Verifier row overhead dominates | duplicate-row K2 loses almost as much as active K2 | optimize verifier rows before candidate work |
| Row0 drift | K2 row0 differs from K1 | keep explicit K1 trunk until batch-invariant proposer is proven |
| Gate overfires | enabled rate high but tps drops | tighten confidence/EMA gate |
| Hybrid state bug | accepted row leaves stale KV/GDN state | block on byte-exact/off correctness and no-copy canaries |

---

## 8. Success Criteria

Minimum ship candidate:

- K1 row0 exactness proven in paired traces.
- `proposer_free=true` on >=99% of enabled row1 events.
- `extra_extend_one_calls=0` in free-row1 mode.
- `generated_rows == verified_rows <= 2`.
- Gated K2 improves decode TPS over K1 in the same runtime window.
- Gated K2 does not reduce accepted tokens/event versus K1.
- Correctness/off comparison remains byte-exact under the existing F_b gates.

Non-goals for this spec:

- K>2 support.
- packed unique-node verifier.
- suffix automaton integration.
- relaxed/typical acceptance.
- SWE-Bench-scale launch before the microbench matrix passes.
