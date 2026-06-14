# FR13 — chain3 floor-probe: DEPTH LEVER DEAD (chain3 = chain5 = 5 flips); flips scale with realization-gap MAGNITUDE, not depth

Date 2026-06-14. GPU workflow `wf_5db71b8b-326` (task w9e398wye), GateChain3 phase. chain3 ran engaged
(tok/draft=3), within_boot_det [T,T,T,T], served full [128,128,128,128]. Raw:
`research/fr13_workflows/chain3_floor_probe_wf_5db71b8b.*`.

## The decisive number: chain3 (depth-3) = 5 clear-margin flips = chain5 (depth-5) = 5
Full like-for-like frontier (each arm vs its OWN no-spec decode oracle, thr 1.0 nat, same prompts_swe4):
| arm | shape | per-prompt clear flips | total | accept/event (cross-boot confounded) |
|---|---|---|---|---|
| native E5 (FLASH MTP-5) | depth-5 | [0,1,1,1] | **3** | 3.076 |
| **chain3** (ours) | depth-3 spine, no width | [0,1,3,1] | **5** | 2.295 |
| chain5 (ours) | depth-5 spine, no width | [0,0,5,0] | **5** | 2.664 |
| cat9 (ours, LOCKED) | depth-5 + 4 leaves | [5,7,4,6] | **22** | 3.198 |

→ **Cutting committed depth 5→3 did NOT reduce flips (5→5). The depth model (flips ≈ 0.6D+0.4D,
predicting D=3→~3) is FALSIFIED.** This confirms the constant-position alternative
(FR13_RESHAPE_DEPTH_MODEL_BIND issue #1): flips do NOT scale with accept-depth.

## The corrected model: flips = monotone(realization-gap MAGNITUDE) over a FIXED boundary set
The clear-margin flips sit at a fixed set of genuine high-entropy decision boundaries (```bash↔```python,
\n↔\n\n, ` Let`↔```` ``` ````, `<tool_call>`↔```` ``` ````, find↔cat — deviations 1.1–9.75 nat). The COUNT
that flip is set by the MAGNITUDE of the arm's per-forward realization gap vs its own no-spec oracle, NOT by
spine depth or the number of rank-1 scan steps:
- native's small MTP-vs-no-spec gap → **3** flips (native's own irreducible floor);
- our pure-spine gap (TREE_ATTN + tree-GDN rank-1 scan vs chunked-prefill) → **5**, CONSTANT across depth 3/5;
- cat9's leaf CO-RESIDENCY (4 deep leaves sharing the forward) → a much larger perturbation → **22**.

## What this means for the GOAL (cat9 22 → native 3)
Decomposes into two INDEPENDENT gaps:
1. **Leaf co-residency = +17** (cat9 22 vs spine 5). REMOVABLE by reshape (leaf-free spine → 5). BUT a
   leaf-free spine gives up the accept edge (chain5 2.66 < native 3.076 — cross-boot caveat) ⇒ no speed
   win ⇒ self-defeating as a deploy shape.
2. **Our-spine-vs-native floor = +2** (spine 5 vs native 3). NOT depth-reducible, NOT leaf-related — it is
   the our-kernel-vs-native per-forward realization gap (the GDN scan state-feed chunk-vs-recurrent
   ~ULP, [[project_fr13_22flip_carrier_l0gdn]], [[reference_diffuse_gdn_accumulation_explained]]).
   Closing it = scan-state bit-exact alignment = **WY territory (PARKED, not revived without the user)**,
   or another alignable seam (UNRESEARCHED — the FA2-fork 2-ULP floor also contributes here).

**So reshape ALONE cannot reach native 3 — its floor is ~5 (the our-spine realization gap). Reaching native
3 requires closing the +2 spine floor (scan-align / WY / another seam).** This is the candidate WALL — but
per [[feedback_research_before_deadend]] do NOT conclude it without first researching whether the +2 is
closable WITHOUT WY (decompose the 5-vs-3: scan state-feed vs FA2-fork 2-ULP vs conv) and whether cat3w's
shallow width avoids the leaf co-residency (the deployable tension).

## OPEN: cat3w (booting now) = the deployable-tension test
cat3w = depth-3 spine + root `(1,)` + d1 `(0,1)` shallow rank-1 siblings. If cat3w ≈ 5 with accept >
chain3's 2.295, shallow width recovers accept WITHOUT the deep-leaf co-residency ⇒ a deployable shape at
the ~5 floor (still +2 over native). If cat3w >> 5, shallow width also adds co-residency. Either way the
spine floor (+2 vs native) stands as the GOAL-blocking gap. Pairs with
[[project_fr13_tree_reshape_unifying_lever]], [[reference_scalar_metric_per_token_blindspot]],
[[feedback_check_artifact_before_concluding]], [[feedback_grind_all_fronts_dont_re_escalate]].
