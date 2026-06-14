# FR13 — tree-reshape EXHAUSTED: depth dead + width adds co-residency; no deployable lossless+fast shape. Two remaining levers: oracle-frame + leaf-co-residency fix

Date 2026-06-14. GPU workflow `wf_5db71b8b-326` (task w9e398wye) complete, verdict holds=True. Raw:
`research/fr13_workflows/chain3_cat3w_wf_5db71b8b.raw.json`. Closes the reshape campaign
([[project_fr13_tree_reshape_unifying_lever]]).

## The complete reshape frontier (each vs its OWN no-spec CHUNKED oracle, thr 1.0 nat — see frame caveat)
| arm | shape | flips | accept/event (cross-boot conf.) | within_det |
|---|---|---|---|---|
| native E5 (FLASH) | depth-5 | **3** | 3.076 | — |
| chain3 (ours) | depth-3 spine, no width | **5** | 2.295 | [T,T,T,T] |
| chain5 (ours) | depth-5 spine, no width | **5** | 2.664 | [T,T,T,T] |
| cat9 (LOCKED) | depth-5 + 4 leaves | **22** | 3.198 | — |
| **cat3w** (ours) | depth-3 spine + root + d1 width | **25** | 2.108 | [T,T,T,T] |

## Two decisive negatives
1. **DEPTH lever DEAD:** chain3 (D=3) = 5 = chain5 (D=5) = 5. Cutting committed depth does NOT reduce flips
   (depth-POSITION crystallization at L60/L61, not accept-depth). (chain3 spreads [0,1,3,1] vs chain5 [0,0,5,0]
   — class-12 distribution shuffle, COUNT unchanged.)
2. **WIDTH ADDS co-residency:** cat3w (D=3 + 2 shallow rank-1 leaves) = 25 vs chain3 (same spine, no width) = 5
   — a ~5× jump from just 2 leaves, WORSE than cat9's 22. Every prompt rose ([4,10,4,7] vs [0,1,3,1]).
   Shallow width is NOT "free." Note: the root `(1,)` sibling is strict-mask attention-invisible to the spine,
   yet cat3w still jumped ⇒ the co-residency is **GDN co-residency** (leaves share the GDN forward →
   batch-variance perturbs the spine state), NOT attention. (cat3w stream is shorter [86,124,128,128] = some
   class-12 trajectory confound, but the +20 direction is far too large to be only trajectory.)

→ **No shape is deployable as BOTH flips~native AND accept≥native.** Leafless spines (chain3/chain5 = 5) are
near the floor but branchless = poor speed; width-bearing shapes (cat9 22, cat3w 25) get accept but the
leaves' GDN co-residency dominates the flips. Reshape cannot resolve the leaf-accept ↔ co-residency-flips
tension. **Reshape EXHAUSTED.**

## The cat9 22 decomposes into TWO INDEPENDENT gaps, with DIFFERENT remaining levers
- **+2 spine floor (spine 5 vs native 3)** = the GDN scan state-feed measured vs the WRONG (chunked) oracle
  frame ([[project_fr13_22flip_carrier_l0gdn]] / FR13_PLUS2_NOT_WALL_ORACLE_FRAME_BIND: q1 = ~1 ULP vs the
  RECURRENT oracle, 9.14× smaller than vs chunked). LEVER = the **oracle-frame decision** (recurrent vs
  chunked) — being hard-red-teamed (wf wi6rcn4v8) for a user-decision package. NOT a kernel fix; NOT WY.
- **+17 leaf co-residency (cat9 22 vs spine 5; cat3w +20)** = the leaves' GDN co-residency perturbing the
  spine within one tree forward. This is the REAL per-forward lossless defect for any accept-bearing shape,
  and it is UNLOCALIZED (the in-process sub-op A/B FR13_GDN_SUBOP_MAB crashed on the reduced-row geometry —
  FR13_SUBOP_AB_CRASHED_PIVOT_CHAIN3_BIND; fixable). LEVER = localize the co-residency sub-op + make the
  spine M-invariant to leaf co-residency (so leaves give accept WITHOUT perturbing the spine).

## Disposition (heading to a user surface, not a self-close)
Reshape is exhausted. The campaign's remaining path to "cat9-shape lossless+fast" is NOT reshape but:
(a) settle the **oracle frame** (does the spine 5 collapse to ~native 3 vs the recurrent oracle? = the +2);
(b) **localize + isolate the leaf GDN co-residency** (the +17; re-fix the crashed sub-op A/B). BOTH are open,
neither is a wall, neither is WY. Surface the complete package (reshape-exhausted + the oracle-frame
lineage-change decision + the co-residency path) to the user once wf wi6rcn4v8 lands the decision doc.
Pairs with [[reference_scalar_metric_per_token_blindspot]], [[feedback_check_artifact_before_concluding]],
[[feedback_research_before_deadend]], [[reference_multispine_not_lossless_closed_nonship]].
