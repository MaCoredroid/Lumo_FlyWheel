# FR13 — branch flips LOCALIZED: SPINE_PERTURBATION via co-residency (batch variance), NOT branch-row verification

Date 2026-06-14. Workflow `w0tokpq9g` (`wf_21fd1f2d-9a4`), Verify **holds=True** (with a mechanism caveat).
Raw: `research/fr13_workflows/branch_flip_fix_plan_w0tokpq9g.raw.json`. Follows chain5 (pure spine = 5 flips
vs cat9 22 => branches carry ~17).

## Localization (DECISIVE, code-pinned)
The 11 channel-2 flip records are 11/11 on the **SPINE** (node_ids 0,1,3,5,7 = best_path; node_type
{spine:9, bonus_self_spine:1, reject_correction:1}); **ZERO on the off-spine leaves** (nodes 2,4,6,8). The
committer is clean (ch1 0 clear-margin violations). So the branches do NOT fail their own verification —
they **perturb the co-resident SPINE rows**. chain5 (same kernel+TREE_ATTN, no branches) = 5 confirms it.
=> the carrier is **co-residency BATCH VARIANCE** on the spine rows, at num_accepted∈{4,5} (deep full-accept
steps where all branches co-reside).

## Mechanism: a TILE-CONFIG / reduction-order artifact (NOT data sharing) — exact op REFUTED on GB10
The proposed op (fp8 block-GEMM nearest-M config lookup `configs[min(...abs(x-M))]`, `fp8_utils.py:1235`,
reorders the fp32 K-accum at different M) is **dead on GB10**: `get_w8a8_block_fp8_configs` returns None
(no Spark JSON) → the default config is already **M-invariant**, that line never runs. So the exact
perturbing op is UNCERTAIN, but the spine-perturbation is empirically real (topology evidence + chain5).
RULED OUT as the channel: TREE_ATTN (strict -inf ancestry mask → spine softmax algebraically independent of
branch keys), GDN scan/conv/gate/o_proj/in_proj (bit-exact, select-by-mask folds no branch into spine).

## vLLM batch-invariant scope on GB10 (the wiring caveat)
`enable_batch_invariant_mode` overrides only `aten::mm/addmm/matmul/linear/bmm/softmax/mean`; the fp8 GEMM
is a CUSTOM op (`w8a8_triton_block_scaled_mm_func`) NOT covered. AND GB10 (sm_121/family-120) is NOT
`is_device_capability_family(100)` → takes the ELSE branch in `enable_batch_invariant_mode` (a reduced
override set). `FR13_BI_TREE_ATTN` (Method-A) IS fully built/wired (off-but-built): appends TREE_ATTN to
decode_invariant_backends + `num_splits=1` under VLLM_BATCH_INVARIANT.

## FIX PLAN — GPU batch-verify (BI is a numerics-pinning diagnostic; lossless-safe; speed cost OK to test)
- **C1 (decisive first): cat9 + BI-on** — `BATCH_INVARIANT=1 FR13_BI_TREE_ATTN=1` (+ locked flags), flag-only
  zero-code. If 22 → ~5: carrier is BI-coverable → BI (or a targeted BI override) is the fix. If stays 22:
  carrier is non-BI-coverable (fp8-GEMM inherent M-dependence or diffuse) → C3/C4.
- **C2 (control): chain5 + BI-on** — should stay ~5 (no branches).
- C3 (large-code, conditional): BI override for the fp8 custom op (fixed reduction order) — only if C1 leaves
  residual tracing to fp8. C4 (fallback): tree-reshape (lean) to de-amplify, gated BI-OFF.
Gate = `fr13_oracle_stream_teacher_force.py --threshold 1.0`, metric total_clear_margin_flips, ref = no-spec
oracle, assert engagement + within_boot_det. Pairs with [[reference_diffuse_gdn_accumulation_explained]],
[[project_l0c_triton_autotune_drift]], [[feedback_no_reroute_reward_hacking]].
