# FR-13 — BUILD the no-copy WY one-pass GDN tree kernel (the lossless+fast endgame)

User decision 2026-06-08: **pivot from grinding the ancestor-replay kernel to BUILDING the WY one-pass kernel.** It clears gap A (lossless BY CONSTRUCTION — per-node output = native-on-path, no per-layer drift grind) AND gap B (removes the +35.8% B=4 GDN-state HBM tax; the replay kernel is 9× HBM and fails speed even at drift 0). This supersedes the replay-kernel L12+ grind (replay stays only as a flag-gated fallback).

## The mechanism (from `FR13_LOSSLESS_FAST_DERIVATION.md`, CPU-validated)
No-copy WY/UT one-pass tree kernel. Per value-head state `S∈ℝ^{d_v×d_k}` (d_k=d_v=128, 16 k-heads / 48 v-heads, GQA 3). Gated delta rule `S_t = g_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ`. The rank-1 reflector product over a path collapses to `∏(I − β_s k_s k_sᵀ) = I − K T Kᵀ` (Schreiber–Van Loan WY; Yang et al. delta rule, arXiv:2406.06484).
- **Each node inherits its parent's compact `(K, T, G)` factor and appends ONE rank-1 reflector (O(1) per node).** Branches share the spine factor up to their fork — no per-node `d_v×d_k` state copy, no ancestor replay. Working set per node = `K[d_k×L]` + `T[L×L]` (~tiny), NOT a 64 KB state.
- **Read-out = native chunk algebra restricted to the node's ancestor path** ⟹ equals the native-on-path oracle (SpecInfer Def 4.1 / STree Eq.4-6 branch-losslessness).
- **Accept-only state commit:** publish only the accepted path's final state in place ⟹ recurrent-state HBM = **1.0× native** (vs replay 9×).

## Sources to revive / reuse
- The abandoned WY microbench kernel: `_tree_gdn_gqa_kernel`, `scripts/fr10_real_dims_tree_vs_fla_cost.py:77-183` (one-pass, capturable — revive for serving).
- The CPU-validated reference + native-op-order basis: `scripts/fr13_lossless_fast_derivation_validate.py`, `output/gdn_novel_research/wy_gated_delta_foundation.py`.
- Wire into the serving GDN tree path by editing `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (replace the `_tree_gdn_kernel` call) + `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`.

## CRITICAL alignment (hold the fp32 4.19e-9 / 0-argmax result on GPU)
1. **Use the `native` basis, NOT the `rescaled` basis** — `rescaled` (k̃=exp(−G)k) blows up to 46573× at depth 64 and is unsafe in fp32. Use the bounded decay-matrix fp32 solve (derivation §1.3).
2. Native op-order on the within-path solve: **fp32 accumulation**, `solve_tril` forward-substitution row order, **`tl.range` static unroll** (not `static_range` rescale), l2norm-in-kernel eps 1e-6, raw-g/softplus gating in-kernel, bf16 conv tap boundaries (already done). Same alignment class that fixed conv (ex2.approx) + the L12 scan-state (tl.range).

## Wiring + flag
Flag-gate the WY kernel (e.g. `FR10_TREE_GDN_WY=1`), default OFF, replay kernel as fallback. WY touches ONLY the tree-verify path — NOT regular decode (Gate-2 must stay 0.0).

## Validation gates (STRICT, bind each to `FR13_LADDER_LOG.md` per commit)
1. **Gate A VERIFY-PATH:** strict top-down ladder (input + every layer + final logits = 0.0, **spine AND branch**) vs native-on-path oracle, fallback UNSET. WY should be lossless by construction; any layer that drifts is an op-order seam to align (same class — fix in-kernel, no copy/splice). Within the bf16/fp32 self-noise floor counts (per the accepted gate), but verify the GDN-scan sub-gate hits the fp32 4.19e-9 / 0-argmax bar the CPU derivation cleared.
2. **Gate 2 REGULAR-DECODE:** forked FA2 no-bias plain decode == pristine = 0.0 every layer (unchanged; WY must not touch regular decode).
3. **Confirm CUDA-graph FULL capture** (WY is static-loop → capturable; the replay's per-forward alloc / unconditional FR12 clones must be env-gated OFF — see `FR13_SPEED_AND_LOSSLESS_GAPS.md`).

## Then e2e (CLEAN, B=4) — the deliverable
`FR10_METRICS=0`, ALL FR12/FR13 diagnostic capture env-gated OFF, B=4, CUDA-graph FULL, same8, temp 0.6/top_p 0.95. Measure vs **E5** (`output/fr10_native_mtp5_same8_20260604T210257Z`, native MTP-5 accept/event 3.21 on this harness):
- **Lossless:** bag-TV vs E5 ≤ self-noise floor (~0.059).
- **Superset:** accept/event ≥ 3.21 (drift=0 ⟹ ≥ native by math).
- **Speed:** per-request decode TPS ≥ native (WY removes the +35.8% B=4 HBM → should now be ≥ native). Report all three; **do NOT self-declare pass/fail — bring numbers to the user.**

## Discipline (standing, user)
ONE GPU (no concurrent docker --gpus; relaunch WITHOUT --rm; `recover_host_memory` / sync+drop_caches between arms; nvidia-smi clean before launch). **NO copy / state-copy / reroute / splice / dense / copy-recurrent** — OUR WY kernel computes, verified vs the native-on-path oracle (splice OFF). `FR10_ALLOW_LINEAR_FALLBACK` runs are DIAGNOSTIC ONLY, never bound to a commit. Commit+push+bind EVERY step (in HEAD AND pushed). Report at the deliverable (lossless + superset + TPS ≥ native) or a genuine un-grindable wall.
