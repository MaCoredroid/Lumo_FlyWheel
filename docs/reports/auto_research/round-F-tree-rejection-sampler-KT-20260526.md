# KT / Handoff: Implement tree-aware rejection sampling in vLLM 0.19.0 to ship F_a (MTP branching-tree spec)

**Generated:** 2026-05-26
**Audience:** the agent(s) picking up the tree-spec verify fix
**Status:** Root cause fully localized and evidenced. Drafting + tree-attention
verify-forward already work; the **acceptance half** (tree-aware rejection
sampling) is missing in vLLM 0.19.0. This doc tells you exactly what's broken,
how it was found, where to patch, the metrics to use, and the pass criteria.
**Deployed stack:** vLLM `0.19.0`, container `lumo-vllm-track-b-suffix`, Qwen3.6-27B
FP8 (multimodal `qwen3_5` checkpoint, **M-RoPE**), GB10 (DGX Spark), TP=1,
realized `kv_cache_dtype=auto` (bf16).

---

## Findings / conclusion (2026-05-26): F_a no-ship on Qwen3.6 GDN hybrid

**Conclusion:** do **not** ship F_a packed branching-tree speculative decoding
for Qwen3.6. The tree-aware rejection-sampler patch is useful as a reference
implementation, but packed tree verification is architecturally capped on this
hybrid model and failed both hard gates. F_a does not beat E3, and the breadth-
first packed tree violates the top-1-path invariant.

**Final root cause:** after adding tree-aware rejection sampling, parent-logit
remapping, and depth-based target M-RoPE positions, the remaining failure is not
the sampler. Qwen3.6 is a hybrid `qwen3_5` model with GDN/GatedDeltaNet sequence
layers. `TreeAttention` masks only softmax self-attention; the GDN speculative
path is linear-chain-oriented and processes the packed tree in flattened order.
For breadth-first trees, sibling nodes are evaluated before deeper top-1
continuations, so the non-attention recurrent/convolutional state corrupts
depth-2/depth-3 top-1 verification logits. A packed tree cannot be made
lossless by attention masks alone.

**Probe evidence (B=1 `spec_speed_probe`, greedy temp 0, 10 fixed prompts):**

| Config | Shape | decode_tps | acc/event | P(acc=3) | Result |
|---|---|---:|---:|---:|---|
| E3 | linear MTP depth 3 | 17.69 | 2.209 | 0.585 | Baseline |
| F_a breadth-first | top-2 root, 6-node packed tree | 16.00 | 1.836 | 0.329 | Fails invariant and speed |
| F diagnostic | contiguous top-1 path + side leaf | 17.17 | 2.140 | 0.613 | Restores near-E3 top-1 behavior but still slower/lower acc |

The decisive comparison is breadth-first versus contiguous top-1. Keeping the
top-1 path contiguous restores acceptance close to E3 (`acc/event=2.14`,
`P(acc=3)=0.613`), while the intended breadth-first top-2-root tree remains far
below E3 (`acc/event=1.836`, `P(acc=3)=0.329`). Since the top-1 path is supposed
to be exactly the E3 chain, breadth-first F_a must be at least E3 if packed-tree
verification is correct. It is not.

**No-ship decision:** F_a packed tree-attention is **not losslessly viable** on
the Qwen3.6 GDN hybrid and does **not** beat E3. Do not sweep more packed-tree
shapes as a shipping path. The only lossless multi-candidate route for this
model family is F_b: verify each candidate path as a separate contiguous
sequence, so GDN/Mamba-style layers see a normal linear history per path.

---

## 0. Goal & success criteria

Ship **F_a** = native-MTP **branching-tree** speculative decoding that is **faster
than E3** (linear MTP, depth 3) on the same workload, losslessly.

**Baseline to beat (B=1 `spec_speed_probe`, temp 0.6, 10 fixed prompts, identical inputs):**

| | E3 (linear, draft=3) | F_a now (tree, draft=6) | F_a target |
|---|---:|---:|---:|
| decode_tps (B=1) | **17.67** | 10.97 | **> 17.67** |
| acc/event | 2.235 | 0.974 | **≥ 2.235 + branch bonus** |
| accepted/node | 0.745 | 0.162 | higher |
| acc_dist | {0:88,1:114,2:115,**3:476**} | {0:141,**1:1049**,2:105,3:1} | mass at depth ≥2 |

**Pass criteria for the fix:**
1. F_a `acc/event` ≥ E3's (2.235) and ideally higher (branches add accepts the
   linear chain can't), `decode_tps` > E3's 17.67.
2. **B-2 lossless still holds**: greedy (temp 0) output **byte-exact identical** to
   config OFF (no-spec) on the fixed prompts. Tree-rejection must preserve the
   target distribution exactly (rejection-sampling theorem).
3. No new EngineCore crash; per-agent trace shows draft=node-count, acc reaching
   depth 3.

---

## 1. The definitive root cause

**vLLM 0.19.0's tree speculative decoding is half-implemented.** The *draft* side
(`propose_tree`) and the *verify-forward* (tree-attention mask) exist and are
**correct** (evidence in §2). But the **rejection sampler is flat-chain — it has
no tree-path awareness** — so branching trees collapse to **acc=1**.

**Code evidence (deployed `/usr/local/lib/python3.12/dist-packages/vllm/`):**

1. `v1/spec_decode/metadata.py` — `SpecDecodeMetadata` carries **no tree/parent
   topology**: only `draft_token_ids` (flat), `cu_num_draft_tokens`,
   `target_logits_indices`, `bonus_logits_indices`. Drafts are flattened
   (`flattened_draft_token_ids = sum(draft_token_ids, [])`), losing tree structure.

2. `v1/sample/rejection_sampler.py`:
   - `rejection_sample()` (~line 350) → `rejection_greedy_sample_kernel`.
   - The kernel walks the flat draft list **sequentially**, accepting a **prefix**
     until the first mismatch:
     ```python
     for pos in range(num_draft_tokens):
         if not rejected:
             if target_argmax[start+pos] == draft[start+pos]:  # accept
             else: rejected = True   # STOP — no descent into a child
     ```
     There is **no parent-pointer / path-following logic**.

3. `propose_tree` output is **breadth-first flattened** (sorted by `(len, tuple)` in
   `config/speculative.py` ~line 594): for our default tree the order is
   `[(0,),(1,),(0,0),(1,0),(0,0,0),(1,0,0)]`.

**Why this gives acc=1 for any branching tree:** the kernel accepts node 0 `(0,)`
(the top-1 first token), then checks node 1 `(1,)` — which is the **sibling**
(top-2 first token), **not** the continuation of node 0. The target's argmax after
`(0,)` is the real next token (= the depth-2 draft, which is *correct*), not `(1,)`
→ mismatch → `rejected=True` → stop. **acc=1, always.** Depth-2/3 nodes are never
reached by the prefix walk.

This is **not** M-RoPE-specific and **not** caused by our patches — it's a core
gap. It also explains why branching trees aren't used in practice on this vLLM.

---

## 2. Evidence the drafts are correct (fix is verify/accept-side ONLY)

Don't waste time on the draft side — it's proven correct:

- **Detokenized top-1-path proposals track the actual output exactly.** With the
  `--tree-debug` logger (§5), on prompt "Write a Python function to reverse a
  linked list" the actual greedy output is
  `"Here's a thinking process:\n\n1.  **Understand the User Request:**"` and the
  tree's per-event top-1 path proposes exactly the next tokens
  (`'s`→` a`→` thinking`, ` thinking`→` process`→`:`, `1`→`.`→` `, …). The depth-2
  and depth-3 draft **tokens are the correct continuations.**
- **Acceptance by depth:** F_a depth-1 accept = **89.1%** ≈ E3's **88.9%** (first
  draft is perfect); depth-2 collapses to **8.2%** vs E3's **74.5%**. The cliff is
  precisely at the point where the flat-chain kernel hits the sibling.
- **B-2 lossless passes** (byte-exact vs OFF): because a rejected-but-correct
  depth-2 token comes back as the **bonus** token (re-sampled identically) — same
  output, lost acceptance. This is why it's lossless-but-slow.

Conclusion: the tree draft + tree-attention verify-forward are correct; **only the
rejection/acceptance step needs tree awareness.**

---

## 3. The fix: tree-aware rejection sampling

Implement tree-path acceptance (SpecInfer / EAGLE-2 / SEQUOIA style). Two layers:

### 3.1 Thread tree topology into the metadata
`SpecDecodeMetadata` (`v1/spec_decode/metadata.py`) needs **per-node parent
indices** (within each request's flattened block). Source of truth:
- The tree is `speculative_config.speculative_token_tree` (a list of node paths,
  e.g. `[(0,),(1,),(0,0),(1,0),(0,0,0),(1,0,0)]`). Parent of node `t` = the node
  whose path is `t[:-1]`; root nodes (`len==1`) have parent = the "root"/bonus slot.
- `eagle.py` already precomputes `tree_choices`, `cu_drafts_per_level`,
  `child_drafts_per_level`, `tree_draft_pos_offsets` (`__init__` ~lines 256-278) —
  reuse these to build a `parents: list[int]` (cf. Arctic's `SuffixDecodingDraft.parents`
  in `arctic_inference/suffix_decoding/cache.py`, which encodes exactly this).
- Populate it in `gpu_model_runner._calc_spec_decode_metadata` (~line 2584).

### 3.2 Tree-path acceptance kernel
Replace/branch `rejection_greedy_sample_kernel` (and the random-sampling path) with
a tree walk:
- **Greedy:** start at the root's accepted token; descend by repeatedly choosing,
  among a node's children, the child whose token == the target argmax at that
  node's logit; accept it; continue until no child matches; the bonus token is the
  target argmax at the last accepted node. This yields the **longest accepted
  root-to-leaf path** — which for a branching tree is ≥ the linear chain's accept
  length, plus wins when a non-top-1 branch matches.
- **Random (temperature > 0):** SpecInfer-style tree rejection sampling — at each
  node, do the standard rejection/recovery test against the target prob, but the
  candidate set at a node is its **children** (multiple), and on rejection you
  sample the recovered token from the residual distribution. Must preserve the
  target distribution exactly (verify with B-2 + KL). See SpecInfer §3.
- Keep the bonus-token logic (`forward()` ~line 60, `bonus_logits_indices`).
- `target_logits_indices` must map each tree node to the logit slot that predicts
  *its* token (i.e., its parent's output position). Verify this mapping is
  tree-correct (it may currently assume a chain).

**Losslessness is the hard constraint.** The accepted path must be distributed
identically to base decoding. Greedy is the easy case (byte-exact); get that
passing B-2 first, then do the random path.

---

## 4. Secondary issue: target-verify positions (handle AFTER the sampler)

Once tree acceptance follows paths, deeper nodes' positions matter. Currently the
**target** assigns **sequential** positions to the flattened tree nodes, not
depth-based:
- non-mrope: `gpu_model_runner` ~line 1998 `self.positions = num_computed_tokens + query_pos` (plain arange).
- mrope: `_calc_mrope_positions` ~line 2527 `get_next_input_positions_tensor(context_len, num_new_tokens)` (sequential).

Correct RoPE position for a tree node = `base + depth(node)` (e.g. our tree →
`base + [1,1,2,2,3,3]`), not `base + [1,2,3,4,5,6]`. For the *top-1 path's* depth-2
acceptance this does **not** bite (the parent node `(0,)` is at `base+1` under both
conventions — which is why depth-2 *should* accept once the sampler is fixed), but
deeper paths and non-top-1 branches need depth-based positions to verify correctly.
The draft side already uses depth-based RoPE (`eagle.py:propose_tree` `draft_positions = positions + (level+1)`),
so the target must match. Fix in the two sites above, threading the tree depth per
scheduled spec token. **Validate with B-2 after.**

---

## 5. Diagnostic toolkit (metrics & logs) — already built

All committed. Use these to reproduce and to verify the fix.

### 5.1 `scripts/spec_speed_probe.py` — the primary instrument (B=1)
Sends N fixed prompts strictly sequentially to the live vLLM (`:9950`, direct,
bypasses proxy), isolates each request's per-event spec-trace rows, and reports
the accept-vs-cost decomposition. Run:
```bash
.venv/bin/python scripts/spec_speed_probe.py --label <cfg> --temp 0.6 --max-tokens 256
# -> output/spec_speed_probe/<cfg>.json + console summary
```
Metrics: `mean_acc_per_event`, `mean_draft_per_event`, `accepted_per_node`
(=acc/draft, the efficiency headline), `wasted_nodes_per_event`, `mean_event_ms`
(clean per-event latency at B=1), `decode_tps`, `acc_dist` (depth profile).
Cross-check baked in: `committed(trace) ≈ completion_tokens` to ~0.1% (validates
the trace + probe). Single-stream by design — compares configs cleanly; not equal
to the B=4 `dgx_steptrace` numbers.

### 5.2 Per-agent spec trace `per_req_spec_trace.jsonl`
Source-edit of `Scheduler.make_spec_decoding_stats` (in
`relaunch_qwen36_round.py` `_SPEC_TRACE_BLOCK`) logs per event:
`{ts, rid, draft (proposed node count), acc (accepted path length), inv (rejected)}`.
Host path `/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl`.
After the fix, `acc` should reach 2-3 for the tree.

### 5.3 `--tree-debug` per-level draft-token logger
`relaunch_qwen36_round.py --config F --mtp 3 --tree-debug` exports
`LUMO_TREE_DRAFT_DEBUG=1`; `propose_tree` logs each level's proposed token IDs +
base position to `/logs/tree_draft_debug.jsonl`. Use to confirm drafts stay
correct after changes (detokenize via the container tokenizer at
`/models/qwen3.6-27b-fp8`).

### 5.4 acc-by-depth analysis
From `acc_dist`: `P(acc≥k)` per config. The decisive signal was F_a depth-1 ≈ E3
but depth-2 collapse. After the fix, F_a's `P(acc≥2)`, `P(acc≥3)` should approach
or exceed E3's (0.745 / 0.60).

### 5.5 B-2 lossless gate (the correctness guard)
Greedy (temp 0, seed 0) byte-exact compare of config-F output vs config OFF
(no-spec; `/tmp/relaunch_qwen36_off.py`, same KEEP prelaunch). Must be IDENTICAL.
Run this after every sampler change.

---

## 6. Reproduce + verify loop

1. **Serve config F (tree):**
   `source .lumo.local.env && export LUMO_SUDO_PASSWORD`
   `.venv/bin/python scripts/swe_x86_helpers/relaunch_qwen36_round.py --config F --mtp 3 [--tree-debug]`
   (~8 min; transient GPU-mem OOM on first `docker run` self-heals via
   `gpu_mem_util` backoff — not a fault. Wait for `READY config=F`.)
   Confirm `[cuda.py] Using AttentionBackendEnum.TREE_ATTN backend` and the
   prelaunch lines `applied tree-attn force patch` + `applied propose_tree M-RoPE patch`.
2. **Canary:** one generation; per-agent trace shows `draft=6` (node count), no
   EngineCore crash. (Pre-fix: `acc` pinned at 1.)
3. **Probe:** `spec_speed_probe --label F_after`. Pre-fix: acc/event≈0.97,
   tps≈11. Target: acc/event≥2.2, tps>17.7.
4. **B-2:** relaunch OFF, greedy byte-exact vs F greedy — must match.
5. **Compare** F_after vs E3 (relaunch `--config E --mtp 3`, probe `--label E3`).

Iterate sampler → probe → B-2 until pass criteria (§0) met.

---

## 7. What's already in place (start from here)

Config F is first-class and the **draft + verify-forward work**; you only need the
**accept** side. Implemented (committed):
- `scripts/swe_x86_helpers/relaunch_qwen36_round.py`:
  - `--config F --mtp N [--tree "<choices>"] [--tree-debug]`; `_default_tree(n)` =
    top-2 root → depth-n chains (n=3 → 6-node tree).
  - `_TREE_ATTN_BLOCK`: prelaunch source-edit of `v1/attention/selector.py` forcing
    `TREE_ATTN` for decoder self-attn on a branching tree (vLLM 0.19.0 ignores
    `VLLM_ATTENTION_BACKEND`; both target-verify and draft need the tree mask).
  - `_MROPE_TREE_BLOCK`: prelaunch source-edit of `v1/spec_decode/eagle.py`
    `propose_tree` — 3 MTP-compat fixes (text-only M-RoPE 1D positions reduction +
    `_set_positions`/`_get_positions`; multimodal `inputs_embeds` via
    `embed_input_ids`; `model_returns_tuple()` guard) + the `--tree-debug` logger.
  - `num_speculative_tokens` = tree **node count** (not depth) in `_mtp_bundle`.
  - `_SPEC_TRACE_BLOCK`: per-agent `{ts,rid,draft,acc,inv}` trace.
- `scripts/run_codex_experiment.py`: `apply_config` routes `--config F`.
- `scripts/spec_speed_probe.py`: the B=1 probe.

**Commits (main):** `3a13fbab` (config F plumbing + tree-attn patch + telemetry),
`12472ba8` (mrope propose_tree crash fix), `538ee75b` (spec_speed_probe),
`01a26bca` (propose_tree mm/tuple/node-count), `6613bbf3` (probe results + findings
+ debug logger). Design + findings: `docs/reports/auto_research/round-F-mtp-tree-fa-fb-design-status-20260525.md` (§6b).

**Key source files to patch (in the container image, then bake into a prelaunch
source-edit like the existing blocks, OR a proper vLLM fork):**
- `vllm/v1/sample/rejection_sampler.py` (kernel + `rejection_sample` + `forward`)
- `vllm/v1/spec_decode/metadata.py` (`SpecDecodeMetadata` — add parent/tree topology)
- `vllm/v1/worker/gpu_model_runner.py` (`_calc_spec_decode_metadata` ~2584;
  positions `_calc_mrope_positions` ~2488/2527 and `self.positions` ~1998)
- `vllm/v1/spec_decode/eagle.py` (`propose_tree` ~936; tree precompute ~256-278) —
  reference for tree structure; drafts already correct.

---

## 8. Constraints & references

**Constraints:** stay FP8 weights, Qwen3.6-27B base, realized `auto`/bf16 KV (note:
TreeAttentionBackend does **not** support fp8 KV — `supported_kv_cache_dtypes =
["auto","float16","bfloat16"]`; this is fine since realized KV is auto). TP=1.
Tree-spec is **lossless by construction** (rejection sampling) → gate = B-1/B-2/B-3
distribution equivalence (byte-exact greedy + KL on a fixed prompt set), no
SWE-Bench re-run needed for correctness; run SWE only for the final speed number.

**Don't repeat these dead ends:**
- `VLLM_ATTENTION_BACKEND=TREE_ATTN` env is **ignored** by vLLM 0.19.0 — force via
  the selector source-edit (done).
- The M-RoPE position theory is **not** the depth-2 cliff cause (parent node
  position is correct under both conventions) — the cliff is the flat-chain
  sampler. Positions are a *secondary* (deeper-node) fix.

**References:**
- SpecInfer (tree-based speculative inference; tree attention + tree rejection
  sampling) — the canonical algorithm to implement.
- SEQUOIA (NeurIPS 2024) — scalable tree verification.
- EAGLE-2 — dynamic draft trees + tree verification.
- vLLM Issue #18327 — "Tree-Attention Support for Speculative Decoding" (the
  feature that added the draft/verify-forward but not tree acceptance).
- Leviathan et al. 2023 / Chen et al. 2023 — rejection sampling losslessness proof.

---

## Appendix A: F_b (batched-paths) — the alternative / fallback plan

If the tree-rejection-sampler fix (F_a) proves too costly or risky, **F_b is the
pragmatic path to "more candidates than linear" that needs NO vLLM core change.**

**Idea:** instead of one packed tree verified with a tree mask, enumerate the
MTP top-k as **K parallel linear paths** (root top-k first tokens, each extended
into its own depth-N chain) and verify them as **K separate batched sequences on
FlashAttention** — sharing the prompt prefix via prefix caching.

**Why F_b sidesteps the bug in this doc:** each path is a **linear chain**, so the
**existing flat-chain rejection sampler accepts it correctly** (the very thing
that fails for trees works per-path). No tree topology, no tree-attention, no
tree-rejection needed. It also stays on the **validated FlashAttention** kernel
(no `TREE_ATTN`, no mrope/tree-verify edge cases).

**Tradeoffs vs F_a (tree)** — from the F_a/F_b analysis (see design doc §2):
- Both amortize the 27B weight read if the K paths are batched into one forward
  (the main spec win is captured by either).
- **Scaling:** tree node count grows **additively** and shares ancestors; F_b's
  **path** count grows **multiplicatively** (`b^d` full-length sequences) and
  **recomputes shared ancestors**. So F_b is great for a *handful* of paths
  (e.g. top-2/top-3 at the root, depth 3) but does **not** scale to wide trees.
- **Correctness surface:** F_b needs a custom proposer (emit K chains from MTP
  top-k) + a batched-path verifier that picks the best accepted path with correct
  cross-path sampling — a real, but **bounded** build (~1-2 wk), reusing the
  working per-path sampler. F_a needs the (riskier) core tree-rejection feature
  but then scales additively.

**Implementation sketch (F_b):**
1. Proposer: branch the MTP head top-k at the root (and optionally 1-2 levels),
   emit K linear draft chains (reuse `eagle.py`'s linear chaining, not `propose_tree`).
2. Schedule the K chains as K sequences sharing the prompt prefix (prefix caching
   dedups the context KV); verify in one batched target forward (FlashAttention).
3. Per-path flat-chain rejection (existing sampler); commit the **longest accepted
   path** (with correct joint sampling to stay lossless — greedy first, B-2 gate).
4. Probe with `spec_speed_probe`; same pass criteria as §0 (beat E3, B-2 lossless).

**Decision guidance:**
- **Tree-rejection (F_a) lands** → prefer F_a: additive scaling, can go wide/deep,
  the cleaner long-term substrate for "way more candidates."
- **Tree-rejection too costly** → ship **F_b** for a narrow top-k (it reuses the
  working sampler and beats linear via first-token diversity), or ship **E3**
  (linear, 17.7 tps, lossless) as the safe baseline.
- The two are not mutually exclusive: F_b can ship first (no core change) while
  the F_a tree-rejection feature is built in parallel.

Task tracking: F_a fix = this KT doc; F_b build = task "Build F_b: batched-paths
spec-decode on FlashAttention" (#6).
