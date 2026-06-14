# FR13 — FA2-tile-carrier OVERTURNED; 22-flip carrier is born at L0 GDN (co-residency), not the forked-FA2 query-tile

Date 2026-06-14. **LINEAGE CHANGE** (report-by-exception): commit `9ad6793f`'s verdict
("forked-FA2 query-tile IS the 22-flip carrier, 1 bf16-ULP/full-attn-layer") is **FALSIFIED**.
Source: QPAD build+gate workflow `wf_5c5985ad-fd1` (task w4o515sg8), its own adversarial verdict
`holds=False / GATE1_ONLY_GATE2_FAILED`, cross-checked against the NODE7-LADDER
(`output/fr13_node7_ladder/ladder_summary.json`).

## The falsification (cleanest possible experiment: fix the named carrier → flips don't move)
`FR13_FA2_QPAD` pads the forked-FA2 query to a fixed N_PAD_Q=64 tile (M-invariant across tree
sizes), lossless-by-construction (CPU oracle 0.0 across M=5/9/10, B=1/3, fp32/bf16, paged).
GPU gate (cat9 + FR13_FA2_QPAD=1, eager, locked pipeline):
- **GATE-1 (in-process MAB A/B, decoherence-free):** named carrier **L31 deep_spine_raw_max_abs
  3.90625e-3 → 0.0** (carrier fixed). 14/16 full-attn layers 0.0 (residuals: L23 2-ULP, L35 1-ULP
  — the M5 pad=59 vs M9 pad=54 suffix-KV layouts still slip).
- **GATE-2 (e2e full-stream flips, the DECISIVE instrument):** `total_clear_margin_flips = 24`
  (per-prompt [5,3,5,11]) vs banked cat9 22 / native 3 — **did NOT drop**. accept/event 2.643
  (below native floor). The 24 are the SAME diffuse deep-accept signature (14/24 dev>3 nat,
  max 11.875) → QPAD did not touch the mechanism. (GATE-2 stream forked — served_lens
  [76,103,128,128] vs [128,128,128,126] — so 24-vs-22 is class-12 trajectory-confounded, but a
  *complete non-response* cannot be a confound artifact.)

## Why the FA2 fix CANNOT be the carrier (the decisive argument, beyond the QPAD null result)
The NODE7-LADDER per-layer divergence ladder (deep-spine node, byte-exact input `input_max_abs=0.0`)
puts the **first-nonzero at L0 `linear_attention` (GDN) = 7.8125e-3 (2 bf16-ULP), cos 0.99996**,
then diffuse to L63. The **first full-attention layer is L3** (max_abs 0.00409) — *downstream* of
L0 GDN. A fix to the FA2 full-attn tiles (L3+) is structurally incapable of removing a divergence
that is already 2-ULP at L0. So the forked-FA2 query-tile M-dependence (real, ~1-ULP, measured by
the MAB in 9ad6793f) is a **downstream correlate, NOT the carrier**. Do **NOT** iterate QPAD
(matching the M=1 decode geometry won't help an upstream-born divergence). QPAD branch
`fr13-fa2-qpad` (030a1c22) stays UNMERGED/archived.

## Consolidated flip frontier (each arm vs its OWN no-spec decode oracle, threshold 1.0 nat — COMPARABLE)
| arm | topology | flips | accept/event (cross-boot, class-12 confounded — directional only) |
|---|---|---|---|
| **native E5** (FLASH, MTP-5) | 5-spine | **3** | 3.076 |
| **chain5** (our kernel) | 5-spine, no branches | **5** | 2.664 |
| **cat9** (LOCKED build) | 5-spine + 4 leaves | **22** | 3.198 |
| cat9 + BI | 9-node | 34 | 3.109 |
| cat9 + QPAD | 9-node | 24 | 2.643 |

**Reading:** chain5 (no branches) ≈ native (5 vs 3); the **branches add ~17 flips via co-residency**
(commit 2fe2c567: 11/11 ch2 flips ON the spine, 0 on leaves = SPINE_PERTURBATION). The co-residency
enters at **L0 GDN**, NOT the FA2 tile (this bind), NOT the GDN scan (proven M-invariant, BV-16
bit-exact), NOT the fp8 GEMM (M-invariant on GB10), NOT BI-fixable (cat9+BI=34 WORSE).

## The strategy this implies (keep cat9 topology for accept; make its spine M-invariant)
If cat9's spine rows compute bit-identically regardless of co-residency (M-invariant), cat9 → chain5's
flip level (~5) while the branches still supply the accept edge (3.198) → clears the flip bar (~native
floor) AND keeps accept. The remaining ~5-vs-3 is the intrinsic our-chain-vs-native difference
(separate, smaller). So the lever is **localize the L0-GDN sub-op where co-residency enters, then
align it bit-exact** (the directive's "SPINE + localizable sub-op → ALIGN BIT-EXACT").

## NEXT (decisive): L0-GDN sub-op in-process A/B (decoherence-free, like the FA2 MAB)
Capture the deep-spine row's L0 GDN sub-ops — **pre_conv → conv1d_out → scan_out → gate_out →
o_proj_out** — at M=10 (cat9 tree) vs M=5 (spine-slice) vs M=1 (decode), on the SAME captured input.
First-nonzero sub-op = the co-residency carrier. **Prime suspect = conv1d prior-window**
([[project_fr13_conv_priorwindow_root]]: conv1d_out diverged 18.375 at num_accepted>1, wrong
bank-row/cols, "fixable wiring at fr10_phase4_patch_vllm_tree_gdn.py:797-818"). Pairs with
[[reference_diffuse_gdn_accumulation_explained]], [[reference_gdn_verify_sequential_dispatch]],
[[feedback_check_artifact_before_concluding]], [[feedback_no_reroute_reward_hacking]].
Fallback if no localizable sub-op = the cat8/cat7 reshape frontier (cat8 boot failed, unfinished).
