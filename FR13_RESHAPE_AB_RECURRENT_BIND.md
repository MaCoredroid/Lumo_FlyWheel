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

## Verdict: WIDTH / leaf-co-residency is the CARRIER (depth ~negligible) — and lossless costs speed
**RED-TEAM CORRECTION (monitor, 2026-06-15): the workflow Verdict labeled this "DEPTH is the lever, WIDTH
refuted" — that is CONFOUNDED.** The clean A/B is at FIXED depth-3: chain3 (d3, NO width)=1 vs cat3w (d3,
+width)=17 → **width adds +16**. At fixed width: cat3w (d3)=17 vs cat9 (d5)=18 → **depth adds +1**. So the
de-cascaded-flip carrier is WIDTH / leaf co-residency (+16), NOT depth (+1). chain3 is lossless because it is
**leaf-free**, not because it is shallow; the Verdict conflated depth-reduction with width-removal (chain3 is
both). This CONFIRMS (not overturns) the banked `+17 leaf-co-residency` decomposition.
- **chain3 (depth-3, leaf-free) = 1 clear flip ≤ native 3 = LOSSLESS** (recurrent frame). No leaves → no
  co-residency forks. The deep-spine accumulation (1.166x/layer, FR13_DIFFUSION_DEEP_DIVE) adds only ~+1-2.
- **cat3w (depth-3 + width) = ~17 ≈ OFF cat9's 18**: the root-sibling/d1 leaves RE-INTRODUCE the
  co-residency / structural-boundary forks (class #12) — the FULL carrier appears at depth-3 already. Audited
  cat3w p3 = genuine multiple INDEPENDENT forks (67-69, 72-76, 88, 122-126 separated by dev=0 re-convergence).
- **BINDING ARBITER (accept/event): BOTH reshaped arms DROP to ~2.27 < native 3.08 < cat9-OFF 3.0-3.198.**
  chain3 is lossless-CLEAN but speed-NEGATIVE (~0.8 fewer accepted tok/event). cat9's SPEED (3.198 > native)
  comes precisely from the LEAVES (width) that make it LOSSY (23). Accept-edge and lossy-ness are the SAME
  leaves — coupled.

## What it REFINES in the banked record (REPORTED to user) — NOT a depth-lever overturn
**FR13_CHAIN3_DEPTH_LEVER_DEAD_BIND is CONFIRMED in its core claim** (width/co-residency, not depth, is the
carrier) and only its NUMBER is superseded: the banked chain3=5 was scored vs **chunked-prefill** (confirmed:
"our pure-spine gap (TREE_ATTN + tree-GDN rank-1 scan vs chunked-prefill) → 5"), an inflated frame; in the
deployment-correct RECURRENT frame **chain3 = 1 ≤ native 3 = LOSSLESS** (the spine floor is BELOW native, not
+2). So the earlier "depth lever dead" intuition was directionally right (depth isn't the carrier) but its
spine-floor magnitude was a chunked-frame artifact. Caveat (#12/#9): chain3=1 is a SINGLE recurrent-frame boot;
cross-boot autotune-fork noise means ~1-3, but {1, native 3} ≤ native and << cat9 23/cat3w 17, so the ORDER
(leaf-free=lossless, +width=+16) is robust. Do NOT over-read the exact "1".

## Strategic consequence (the live front)
Reshape gives a lossless shape (chain3, leaf-free) OR a fast shape (cat9, leaves), NOT both — topology alone
can't break the tension because the leaves ARE both the accept edge AND the co-residency carrier. **The only
lossless+FAST path is to KEEP cat9's leaves (the 3.198 accept edge) and CUT the leaf-co-residency divergence
via a kernel fix** (the no-copy / isolated-forward problem the leaves' shared GDN scan creates). K1 tests one
slice of this — whether matching the scan state store-boundary (the path the leaf co-residency flows through)
cuts it. That is what **K1** tests (workflow
`waao62oj0`, committed b91c1bc0): the per-node bf16 `b_h` store-boundary round-trip applied to every GDN layer,
keeping cat9 geometry — does it cut the depth-accumulation carrier the reshape just localized? If K1 drops
cat9 flips toward native at unchanged accept 3.198 → lossless+fast. If not → the tension is fundamental:
relax to accept/event-parity (cat9 fast-but-lossy) OR ship chain3 (lossless-but-slow). Links:
[[project_fr13_tree_reshape_unifying_lever]], [[reference_diffuse_gdn_accumulation_explained]],
[[reference_multispine_not_lossless_closed_nonship]], [[reference_scalar_metric_per_token_blindspot]].
