# FR13 — Reshape A/B (recurrent-oracle frame): DEPTH lever LIVE (lossless shape exists) but SPEED-NEGATIVE; WIDTH refuted

Date 2026-06-15. GPU workflow `w874nsww3` (wf_0e61765e-f88), ReshapeBoot→Verdict, **verify HOLDS=True**.
Raw: `research/fr13_workflows/fr13_reshape_ab_raw.json`. User chose this test (option 1).

## Non-vacuity PROVEN (independently confirmed by the verify agent)
- **Reshape applied**: cat3w tok/draft=5.0==len(TREE)=5, chain3 tok/draft=3.0==len(TREE)=3 (NEITHER is cat9's
  9). Boot driver fails-loud if tok/draft≠len(TREE); patcher exact-match guards `_fr10_is_cat3w/_fr10_is_chain3`
  (fr10_phase4_patch L10744-10772) with a disengagement RuntimeError; no linear-fallback warning in either log.
- **Oracle engaged**: RECURRENT_PATH_ENGAGED=True, `_forward_core_decode_non_spec` calls cat3w=42240,
  chain3=48768, OFF=41376, native=48768. **Frame IDENTICAL across all 4 arms** (FLASH_ATTN, threshold_nat=1.0,
  seed=1313, top_k=20, same model) → baseline reuse legitimate, SAME-FRAME comparison.
- within-boot det [T,T,T,T] both arms. Only TREE varied (locked pipeline flags identical); reward-hack CLEAN.

## The numbers (all SAME recurrent-oracle frame)
| arm | shape | clear-margin flips (raw / de-cascaded) | per-prompt | accept/event |
|---|---|---|---|---|
| native-E5 | depth-5 spine (linear MTP-5) | 3 / 3 | [0,0,2,1] | **3.08** |
| cat9 (LOCKED) | depth-5 spine + 4 leaves | 23 / 18 | [5,4,5,9] | **3.198** |
| **chain3** | depth-3 spine, NO width | **1 / 1** | [0,0,1,0] | 2.266 |
| cat3w | depth-3 spine + root sib + d1 | 27 / ~17 (16-19, audit-fuzzy) | [5,3,3,6] | 2.282 |

## Verdict: DEPTH is the flip lever (CONFIRMED), WIDTH is not (REFUTED) — but lossless costs speed
- **chain3 (depth-3, leaf-free) = 1 clear flip ≤ native 3 = LOSSLESS** in the deployment-correct recurrent
  frame. Removing the deep spine removes the diffuse cross-layer accumulation (the 1.166x/layer growth from
  FR13_DIFFUSION_DEEP_DIVE has fewer layers of runway / the leaf-free spine has no co-residency).
- **cat3w (depth-3 + width) = ~17 ≈ OFF cat9's 18**: the root-sibling/d1 leaves RE-INTRODUCE the
  co-residency / structural-boundary forks (class #12). Audited cat3w p3 = genuine multiple INDEPENDENT forks
  (clusters 67-69, 72-76, 88, 122-126 separated by dev=0 re-convergence), NOT one undercounted cascade.
- **BINDING ARBITER (accept/event): BOTH reshaped arms DROP to ~2.27 < native 3.08 < cat9-OFF 3.0-3.198.**
  chain3 is lossless-CLEAN but speed-NEGATIVE (~0.8 fewer accepted tok/event). cat9's SPEED (3.198 > native)
  comes precisely from the depth+width that makes it LOSSY (23). The two are coupled.

## OVERTURNS a banked closure (REPORTED to user)
This SUPERSEDES **FR13_CHAIN3_DEPTH_LEVER_DEAD_BIND** ("chain3=chain5=5, depth lever dead, flips don't scale
with depth"). In the deployment-correct RECURRENT oracle frame, **chain3 = 1 ≤ native 3 = LOSSLESS**; the
banked chain3=5 was a stale-frame number (2026-06-14, pre-recurrent-oracle — likely the chunked/prefill
teacher-force frame, FR13_ORACLE_FRAME ~9x chunk-vs-recurrent at L0). The DEPTH lever is LIVE: depth+width
co-residency IS the flip carrier, not a "depth-independent floor." Caveat (#12/#9): chain3=1 is a SINGLE fresh
boot; cross-boot autotune-fork noise (the GB10 "no cross-boot byte-identity" floor) means the point estimate
is ~1-3, but ALL of {1 fresh, 3 native} ≤ native and << cat9 23, so the ORDER (chain3 lossless, cat9 lossy,
width re-adds) is robust. Do NOT over-read the exact "1".

## Strategic consequence (the live front)
Reshape gives a lossless shape (chain3) OR a fast shape (cat9), NOT both — topology alone can't break the
tension. **The only lossless+FAST path is to KEEP cat9's topology (depth+width = the 3.198 accept edge) and
CUT the per-layer/co-residency divergence via a kernel fix.** That is EXACTLY what **K1** tests (workflow
`waao62oj0`, committed b91c1bc0): the per-node bf16 `b_h` store-boundary round-trip applied to every GDN layer,
keeping cat9 geometry — does it cut the depth-accumulation carrier the reshape just localized? If K1 drops
cat9 flips toward native at unchanged accept 3.198 → lossless+fast. If not → the tension is fundamental:
relax to accept/event-parity (cat9 fast-but-lossy) OR ship chain3 (lossless-but-slow). Links:
[[project_fr13_tree_reshape_unifying_lever]], [[reference_diffuse_gdn_accumulation_explained]],
[[reference_multispine_not_lossless_closed_nonship]], [[reference_scalar_metric_per_token_blindspot]].
