# FR13 — the +2 spine BV/warps fix is REFUTED (re-derives the already-refuted geometry seam); the +2 is a class-12 cascade (~2 independent events ~ native 3)

Date 2026-06-14. CPU +2-align research `wf_84ac371d-60d` (task w9dpm62la), verify holds=True BUT the verified
fix is REFUTED by a prior empirical bind it missed. Raw: `research/fr13_workflows/plus2_align_research_w9dpm62la.raw.json`.

## The proposed fix (BV 16->32 / num_warps 8->4 on our _tree_gdn_kernel) is a DEAD END
The research claims the +2 spine carrier = our scan Triton codegen (BV=16/warps=8 vs native BV=32/warps=4 =>
different warp-level tl.sum reduction trees => ~1 bf16-ULP/node). This is a STATIC code-reading hypothesis
that **FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND already EMPIRICALLY REFUTED** (RAW max_abs, int-view not atol, L0
GDN, bit-identical inputs, vs the REAL native fused_sigmoid_gating): D16 (ours BV=16/w8) AND D32 (ours
BV=32/w4) = **0.0 at N_PAD=1 AND N_PAD=16**. geometryIsTheSeam=FALSE; our scan is ALREADY bit-exact to native
at both geometries and both tree sizes; "the DIM_K=128 reduction is geometry-stable; the silicon refutes the
static hypothesis." The research's cheap test (fr13_gdn_scan_warp_gate.py BV A/B) would just re-confirm 0.0.
The +2 research's Verify holds=True is WRONG on the fix (it did not check FR13_BV_GEOMETRY). DO NOT pursue
BV/warps alignment. (Pattern: [[feedback_check_artifact_before_concluding]] / [[feedback_math_correct_vs_bitexact]]
inverse — here a static seam was re-proposed after silicon already showed 0.0.)

## What STANDS: the localization (the +2 is a class-12 cascade, not 5 independent flips)
chain5's 5 clear-margin flips are ALL in prompt 2: pos 25 (tool/bash format fork, dev 2.75) -> 26,27,28 are a
CASCADE (teacher-forced against an oracle conditioned on the now-diverged served prefix => read as flips but
only ONE independent decision diverged) + pos 43 (brace order). So 5 raw flips = ~2 INDEPENDENT divergence
events. Native's 3 are one-per-prompt at different boundaries. So in INDEPENDENT crossings our-spine ~2 vs
native ~3 = at-or-below native; the raw +2 is a cascade-inflation (class-12, [[reference_scalar_metric_per_token_blindspot]]).
The honest +2 arbiter = e2e accept/event, NOT the raw flip count.

## Disposition
The scan is bit-exact (math line-for-line identical AND geometry 0.0); conv done; FA2 inert on a branchless
spine; fp8 M-invariant. So the residual +2 (if any, after de-cascading) is genuinely diffuse + small + likely
a cascade artifact. PARK the +2 (do NOT chase BV/warps; revisit via e2e accept/event AFTER the +17 leaf
co-residency fix lands). The PRIORITY remains the +17 (bf16 in_proj_ba M-keyed, the ba-proj implement+test).
If, after the +17 fix, cat9's e2e accept/event >= native, the +2 is moot. Pairs with
[[reference_gdn_kernel_lineage_table]] (FR13_BV_GEOMETRY row stands), [[feedback_research_before_deadend]].
