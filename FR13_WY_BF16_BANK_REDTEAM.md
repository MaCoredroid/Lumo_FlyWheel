# FR13 WY offline bf16-bank probe — red-team (monitor, 2026-06-08)

Source: `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_bf16_bank_batch_best.json`
(schema `fr12.scan_batch_invariance_probe.v1`, `--use-wy --fla-bf16-boundaries --max-depth 6`,
native ref `vllm.fused_sigmoid_gating_delta_rule_update`, build = state-fix `8a975837`).

## VERDICT: state fix WORKED + kernel is order-invariant; residual is the WY-vs-native-FLA reduction-order FLOOR (not a loose op-order seam)

### 1. State fix (8a975837) confirmed — 1.66e-3 → 2.98e-8 fp32
`original_spine_vs_native_fla.state.max_abs = 2.98e-8` (mean 4.1e-12). The separate raw-fp32
`_state` track drives the recurrent carry to the fp32 floor. The S1 hypothesis (bf16-bucket
flip of the carried bank) is **CONFIRMED but tiny**: `state_bf16_bank.mismatch_count = 58 /
4,718,592` (spine-only: 8), `torch_equal = False`, but the **pre-round fp32 delta = 1.33e-12**
(`pre_round_abs`). That is fp32 *noise*, not a tappable op-order gap — you cannot tighten below
it without replicating native's exact chunked fp32 accumulation order (chunk_delta_h.py b_h accum).

### 2. WY kernel is traversal-order INVARIANT (rules out S2/S3 cross-branch wiring)
`reverse_sibling_dfs_full_spine_vs_original_full_spine`: out 0.0, state 0.0, bf16_bank
`torch_equal = True` (0 mismatch). `spine_first_full_spine_vs_original_full_spine`: out 1.49e-8
(1 elem), state 0.0, bf16_bank `torch_equal = True`. → the kernel gives **identical** results
regardless of DFS layout/parent-remap. No off-by-one/ordering bug in the kernel. S2/S3
(cross-branch accepted-column wiring) are NOT kernel bugs — if a live wiring seam exists it is in
the launch harness (`fr10_phase4_patch_vllm_tree_gdn.py`), not `_tree_gdn_wy_kernel`.

### 3. Output is at 1 bf16 ULP
`out.max_abs = 1.22e-4` (1 bf16 ULP, 15898 nonzero, mean 1.09e-7). This is the WY one-pass
readout vs native's chunk_o two-term bf16 split — a **different reduction order** (cascade-map #6).
ℝ-correctness is intact (fp32-oracle path ~4.19e-9). 1.22e-4 is NOT the absolute bf16 floor —
native-vs-native is 0.0; it is the WY-decomposition-differs-from-native-FLA floor.

## What this means (decisive)
- The **"3.32 final logits" was measured PRE-state-fix**. The decisive open question is whether the
  POST-fix build (`8a975837`) is lossless LIVE. Offline is necessary-not-sufficient (depth-1 1.53e-5
  vs live L1 1.22e-4) → the LIVE cumulative ladder is the verdict, not another offline probe.
- Two residual floors remain vs native FLA, BOTH "different reduction order, not loose op-order":
  (a) state 58 bf16-bucket flips @ 1.33e-12 pre-round; (b) output 1.22e-4 = 1 ULP (#6).
- **#6 is the structural user-decision wall:** making the WY output byte-exact 0.0 to native
  requires replicating native's chunk_o two-term readout order (algorithmic, not a tap). Whether
  that is needed depends on the LIVE ladder: if post-state-fix final-logit drift is within E5 floor
  (~0.059 bag-TV) + accept/event >= 3.21, the floor is lossless-enough and #6 is moot. If not,
  #6 (and/or matching native's state chunk-accum order to kill the 1.33e-12) is the next front.

## NEXT (decisive, GPU): ONE live cumulative ladder on 8a975837
Capture-once-native (pin `output/fr13_wy_gateA_*/native/`), re-run only the tree arm, B=1 eager,
FR12 hooks OFF, strict (fallback UNSET): input -> every GDN + full_attn layer -> final logits,
spine AND branch (4 leaf-path oracles). Does post-state-fix final-logit drift fall within E5 floor?
Bind to FR13_LADDER_LOG.md. Then Gate-2 + clean B=4 e2e vs E5.
