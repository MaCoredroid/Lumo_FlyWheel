# FR13 — conv is NOT the carrier (front already done); the 22-flip carrier is the GDN scan STATE-FEED (chunk-vs-recurrent), depth-scaled

Date 2026-06-14. CPU conv-mechanism workflow `wf_d27ad02e-c12` (task wnal576s4), **verify holds=True,
wouldFixReduceFlips=False**. Raw: `research/fr13_workflows/conv_mechanism_wf_d27ad02e.raw.json`.
Reconciles with FR13_GATEA_DEEP_DIVERGENCE.md + FR13_NODE5_LADDER_DIFFUSE_BIND.md (both prior, dual-verified).

## The conv1d is the L0 first-nonzero ENTRY op but NOT the flip carrier
- The tree-conv is **row-occupancy M-invariant BY CONSTRUCTION** (per-b Python loop, no cross-row reduction
  in tap-acc or window gather) — UNLIKE the forked-FA2 kBlockM=64 query-tile. Branch co-residency does NOT
  perturb the deep-spine conv arithmetic.
- The num_accepted-driven prior bank/column selection (the 2026-06-09 conv1d_out=18.375 wrong-bank-row root)
  is **FIXED + STALE on HEAD** — live-confirmed correct because h0_state_in=0.0 co-located with conv1d_out≠0
  at the carrier event (same bank row read; only the MAC differs). **Do NOT re-chase it.**
- The only surviving conv delta = the bf16-tap MAC + ex2-silu realization vs native's fused
  causal_conv1d_update — and per FR13_GATEA "CONV FRONT DONE / the conv silu grind SUCCEEDED" this is
  **already the live path** (triton_ex2_silu_bf16 + bf16-tap arm); the remaining gap is **at most sub-ULP**.
  The new FR13_CONV_FIX_DESIGN.md (b6c30b4b) largely RE-DERIVES the landed replica — low marginal value;
  do not expect it to move the e2e flips (the workflow's own adversarial verdict: wouldFixReduceFlips=False).

## THE CARRIER (3 binds agree) = GDN recurrent SCAN STATE-FEED, chunk-vs-recurrent, depth-scaled
At num_accepted=4 the **live arm builds node-5's state via a rank-1 tree-scan over the accepted chain
[0,1,3,5] seeded from b_h0**; the **clean (no-spec decode oracle) arm builds the same logical state via a
1687-token chunked-prefill scan**. Two realizations of the same recurrent state = the documented
**chunk-vs-recurrent ~1-ULP gap**, born at L0, amplified ~32× by the gate 1/rms + deep full-attn layers,
crystallizing at L60/L61 (FR13_NODE5_LADDER:53-61). The FA2-fork full-attn layers (2-ULP floor) **amplify,
do not originate** (FR13_FA2_CARRIER_OVERTURNED_BIND confirms FA2 is downstream). FR13_GATEA:204: the live
first-divergence AFTER the conv replica succeeded is **h0_state_in = 0.0007 at L8** — a separate ~1-ULP
value-dependent in the GDN recurrent scan state. The scan KERNEL is M-invariant (BV-16 D16=D32=0.0,
single-forward); the STATE-FEED realization across the co-resident accepted chain is the gap.

**Reconciles "branches add 17 flips" (chain5 5 → cat9 22):** the carrier scales with **accept DEPTH**
(longer accepted rank-1 chain = more chunk-vs-recurrent accumulation), and cat9 accepts deeper (3.198) than
chain5 (2.664). Whether the cat9-vs-chain5 excess is pure accept-DEPTH (intrinsic) or also branch
CO-RESIDENCY (an M-dependent op) is the exact disambiguation the concurrent GPU sub-op A/B (w68z6gxgy)
settles: its M10-vs-M5 deep-row first-nonzero — if scan_out M10-vs-M5 ≈ 0.0, it's depth-intrinsic (no
alignable co-residency op → reshape/scan-align); if nonzero, there is a branch-occupancy op to align.

## Levers (do NOT escalate; this is the front map, not a wall)
- **(a) scan state-feed bit-exact align** = make the rank-1 tree-scan match the chunked-prefill realization
  within native's self-floor. This is WY-kernel territory — **WY is PARKED** (failed abs-0.0, not the
  within-floor bar) and is NOT revived without the user. Non-WY sub-levers (fp32 state accumulation, op-order
  / l2norm / raw-g alignment in the rank-1 scan) are open and grindable per FR13_GATEA's "grind all GDN
  fronts" policy.
- **(b) tree-reshape** ([[project_fr13_tree_reshape_unifying_lever]]) = shallower committed spine (less
  depth-accumulation = fewer flips) + root-sibling width (recover the d0/d1 accept WITHOUT deep accumulation).
  The directive's preferred lever; the cat8/cat7 frontier is INCOMPLETE (cat8 boot failed). The tension to
  resolve: shallow reduces flips but also accept (chain5 = 5 flips / 2.66 accept) — root-sibling width is the
  proposed accept-recovery.

## NEXT
GPU A/B (running) confirms the sub-op + co-residency-vs-depth. In parallel (CPU): design the reshape
frontier shapes (shallow-spine + root-sibling) that minimize depth-accumulation while keeping accept ≥
native, predicting each from the depth-accumulation model, to prep the next GPU reshape boot. Pairs with
[[project_fr13_22flip_carrier_l0gdn]], [[reference_diffuse_gdn_accumulation_explained]],
[[reference_gdn_verify_sequential_dispatch]], [[feedback_grind_all_fronts_dont_re_escalate]],
[[feedback_check_artifact_before_concluding]] (the cat9-vs-chain5 accept is cross-trajectory; the flip
COUNT each-vs-own-oracle is the comparable metric).
