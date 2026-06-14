# FR13 — the +17 leaf width co-residency carrier = the bf16 in_proj_ba GEMM (M-keyed); fix = LUMO_FB_BATCH_INVARIANT_BA_PROJ (real, authorized)

Date 2026-06-14. CPU width-locate workflow `wf_065dc1f8-11f` (task w6rbv6qot), **verify holds=True**. Raw:
`research/fr13_workflows/width_coresidency_locate_w6rbv6qot.raw.json`. Doc: FR13_WIDTH_CORESIDENCY_HYPOTHESES.md.

## H1 (PRIME, decisive code evidence) — the bf16 `in_proj_ba` GEMM is M-keyed
When leaves co-reside (M=tree_n=10 vs spine M=5), the bf16 `in_proj_ba` projection produces a ~1-bf16-ULP
shift in **a/b (= the scan's raw_a/raw_b → gate b_g/b_beta)** on the SPINE row → the proven-bit-exact scan
consumes M-variant inputs → the spine output flips. Evidence (verified live):
- `/models/qwen3.6-27b-fp8/config.json`: `linear_attn.in_proj_a` + `in_proj_b` ARE in `modules_to_not_convert`
  (so **bf16 cuBLAS/cuBLASLt**, NOT fp8); `in_proj_qkvz` is NOT (fp8, M-invariant). gdn_linear_attn.py:303
  "ba_proj doesn't support blockwise fp8 quantization."
- bf16 cuBLASLt heuristic + Split-K kernel selection differs M=10 vs M=5 (Thinking Machines #42960:
  small-M/N → Split-K → altered reduction order + tensor-core tile). The ONLY GDN GEMM not proven
  M-invariant, AND upstream of the (M-invariant) scan, AND on the spine data path.
- Existence proof it is M-variant: `LUMO_FB_BATCH_INVARIANT_BA_PROJ` (live gdn_linear_attn.py:553-601) was
  PURPOSE-BUILT to pad the ba projection to a fixed row group "so the BA projection shape is independent
  across active K" — and it is OFF in the locked launcher.

## Refuted alternatives (all with evidence)
- **Cross-branch mask bleed (H4): REFUTED.** fr10_gdn_tree_kernel.py:459-467 ancestry mask folds NO leaf into
  a spine node (strict_mask[spine,leaf]==0 never selected; static_range reduction over masked-to-0 slots);
  FR13_BV_GEOMETRY = scan 0.0 vs native at N_PAD=1 AND 16. The +17 is NOT a mask bug.
- fp8 in_proj_qkvz / o_proj (H3): M-invariant (single M-tile ≤64, K=128 pinned, no Spark cfg). conv (H2):
  row-occupancy M-invariant + consumes fp8 q/k/v not a/b. chunk-boundary (H5): no M-keyed row chunking.
  state-feed/bank (H6): spine reads its own committed bank (h0_state_in byte-exact, geom-fix 8cdda4c4).

## CRITICAL: the current FR13_GDN_SUBOP_MAB A/B is BLIND to H1 (abDiscriminates=False)
The harness receives the ALREADY-projected a,b (post-in_proj_ba at M=tree_n, patcher :3907) and every arm
SLICES them (a0=a[start:end] :1423-1424) → pre_conv/conv1d_out/scan_out all consume the SAME M=10 in_proj_ba
output → **all predict ~0.0 even if H1 is true.** DO NOT run the existing 3-arm A/B expecting it to find H1
(it would produce a vacuous ~0.0 and be MISREAD as "H1 refuted"). It must be EXTENDED with an in_proj_ba
RE-RUN-at-reduced-M arm (re-invoke the projection from a captured PRE-projection hidden span), OR use the
targeted flag test below (preferred).

## The fix = LUMO_FB_BATCH_INVARIANT_BA_PROJ — REAL, not a reward-hack (rewardHackCheck passed)
Pad in_proj_ba to a FIXED, tree_n-independent M (constant row group), compute, scatter real rows back,
discard pads → pins cuBLASLt to ONE shape → M-invariant a/b. Mathematically the SAME per-row computation
(GEMM row = W @ hidden[row], independent per row; zero pads contribute nothing) — does NOT copy/dense/splice
the spine's a/b from a clean run. Batch-invariance is **directive-authorized (#42960)**. Reconciles the prior
cat9+BI=34 (counterproductive): FULL vLLM BI takes the GB10 REDUCED override branch (partial + perturbs the
fp8/scan) → net worse; the TARGETED ba-proj pad ONLY fixes the carrier → should be clean.

## DECISIVE TEST + FIX (one boot) = cat9 + LUMO_FB_BATCH_INVARIANT_BA_PROJ
Wire the env passthrough (locked launcher passes only BATCH_INVARIANT=0; add -e LUMO_FB_BATCH_INVARIANT_BA_PROJ
+ LUMO_FB_KERNEL_ROWS + a fixed pad ≥ max tree_n) → boot cat9 → flip count vs the cat9 oracle.
**cat9 22 → ~5 ⇒ H1 confirmed AND the +17 leaf co-residency is FIXED** (lossless-by-construction: padding
doesn't change real rows; verify det [T,T,T,T] + default-OFF still 22). Stays 22 ⇒ flag insufficient / H1
wrong → the extended in_proj_ba re-run A/B. Pairs with [[reference_diffuse_gdn_accumulation_explained]],
[[project_fr13_22flip_carrier_l0gdn]], [[feedback_no_reroute_reward_hacking]],
[[reference_gb10_gdn_backend_fla]], [[feedback_check_artifact_before_concluding]] (the cat9+BI=34 reinterpret).
