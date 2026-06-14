# FR13 — tree-reshape depth model + ranked shapes (PROMISING but UNVALIDATED; gated by the GPU A/B + a chain3 floor-probe)

Date 2026-06-14. CPU reshape-design workflow `wf_3210769d-aab` (task w3gpipsx2), **verify holds=True**
(model honestly labeled + shapes/feasibility sound; the depth FIT is NOT validated). Raw:
`research/fr13_workflows/reshape_design_wf_3210769d.raw.json`. Doc: FR13_RESHAPE_SHAPE_DESIGN.md.

## Depth-accumulation model (PREDICTION, 1 clean anchor — do not read as fact)
`flips(D) ≈ 0.6·D (native chunk-vs-recurrent floor) + 0.4·D (our-kernel un-aligned-seam excess)`.
Anchors: native E5 (D=5) = 3; chain5 (our kernel, D=5) = 5; cat9 (D=5 + 4 leaves) = 22. Leaf-width nodes
are NEVER fed into the forward/recurrent state (patcher :10955) → add ZERO to the depth term, so **cutting
committed SPINE DEPTH is the flip lever, not trimming leaves**. Predictions: D=3 → ~3 (native regime),
D=4 → ~4, D=5 → 5 (measured). **CAVEAT (Verify issue #1):** the fit has ONE clean pure-spine point
(chain5); the D=3→~3 prediction is EQUALLY consistent with a CONSTANT ~3 native floor that crystallizes at
FIXED positions (L60/L61 deep full-attn = a depth-POSITION effect, FR13_NODE5_LADDER), in which case
cutting depth would NOT reduce flips. **Decisive test = the chain3 floor-probe (does pure D=3 give ~3?).**

## Floor (Verify floorRedTeam): real chunk-vs-recurrent excess; reshape reaches ~3, cannot beat it
chain5's 5 vs native's 3 = a real ~+2-flip our-kernel-over-native excess at D=5 (un-aligned-seam
chunk-vs-recurrent gap, NOT depth-reducible to zero). Reshape's MINIMUM ≈ 3 (a D=3 chain) IF the depth
model holds — coincides with native's 3, clears the within-floor bar (per-depth-argmax + bag-TV ≤ floor,
NOT abs-0.0). It canNOT go below ~3 without scan-alignment (WY, PARKED). "Is reshape's min above native 3?
UNPROVEN, likely yes for any USEFUL (accept-bearing) shape." So reshape is a candidate to REACH native
level, not a guarantee.

## Ranked shapes (each-vs-own-oracle flip count is the comparable metric; accept is class-12 confounded)
Sweep order (GPU serialized): **chain3 (floor probe) → cat3w (rank 1) → cat4w (rank 3) → [cat10-gated]**.
- **chain3** `[(0,),(0,0),(0,0,0)]` (3 nodes, D=3, no width): floor-probe CONTROL. Predicted ~3 flips,
  accept LOW. Answers "does D=3 hit native-3 flips?" — the decisive depth-vs-constant-position test.
- **cat3w** `[(0,),(1,),(0,0),(0,1),(0,0,0)]` (5 nodes, D=3, root+d1 rank-2 siblings): RANK-1 DEPLOY
  candidate. Predicted ~3 flips; accept recovery via shallow width (d0/d1 rescue, cat10 measured d0
  +0.035). accept ≥ native = the RISK axis, UNPROVEN.
- **cat4w** `[(0,),(1,),(0,0),(0,0,0),(0,0,1),(0,0,0,0)]` (6 nodes, D=4): predicted ~4-5, higher accept.
- **cat10-gated** (10 nodes, D=5, root sibling): LOSSLESS-NEGATIVE (predicted FLAT 22 — doesn't cut
  depth); kept only as the d0-rescue building block to LAYER on a depth-cut shape.

## Buildability (verified vs code — NOT a TREE-env-only change)
Drafter topology dispatch is HARDCODED per exact shape (exact-match :9929-9950): only {chain5 (num_spec 5),
cat9 (num_spec 9)} run no-code on HEAD, {cat10} on origin/fr13-cat10-archive. ANY other shape →
disengagement RAISE (:10834) (this is why the earlier cat8 TREE-override boot failed). Each novel shape
needs a **~15-30 line drafter packing branch** (exact-match guard + torch.stack in (len,path) order),
flag-gated/default-cat9-preserving. Downstream (parent/ancestry masks, committer path enum, eager-pack
replay, conv-fusion prior windows) ALL auto-adapt off SPEC_CONFIG tree_choices — ONLY the drafter packing
is hand-rolled. Root sibling `(1,)` CONFIRMED buildable (cat10). **rank≥2 (top-3 fan) NOT buildable** (every
leaf is topk(2)[:,1] — only the rank-1 runner-up is captured); rank-2 width (root/d1/d2 siblings) IS.
**NEVER enable FR10_ALLOW_LINEAR_FALLBACK (banned).**

## The directive's FREE confidence-gated root branch (accept recovery, no retrain)
Emit `(1,)` ONLY when the root top-2 margin is a near-tie. `margin = top1_logit - top2_logit` is FREE —
`_fr10_top2 / _fr10_root_leaf_token = topk(_fr10_logits,2).indices[:,1]` is ALREADY computed; a scalar
compare, no extra lm-head read. Keeps the +0.035 d0 rescue on near-ties (62% of step-0 rejects are 2-horse
near-ties), drops the sibling on confident roots (where cat10's unconditional sibling only diluted d1-d4 →
its net -0.27). Makes draft-toks/event VARIABLE → the class-9 engagement gate must assert tok/draft in the
realizable SET, not a fixed int (or boot ungated for the clean flip number first).

## Gating + next (do NOT escalate)
**The GPU L0-GDN sub-op A/B (w68z6gxgy, running) is the DISAMBIGUATOR — settle co-residency-vs-depth before
trusting the depth model.** Decision rule: (A) A/B finds cat9 spine row diverges from the M5 spine-slice at
a localizable op → the +17 is real co-residency → prefer NO-width or sparse confidence-gated-width shapes
(chain3 then minimal cat3w); (B) A/B finds M10≈M5 on the deep spine row → the +17 is accept-depth/trajectory
→ the depth model is plausible → build + boot the chain3 floor-probe to confirm D=3→~3 (vs the
constant-position alternative). The accept gate MUST use a same-boot native reference (3.16 is a shifting
cross-boot baseline). Pairs with [[project_fr13_22flip_carrier_l0gdn]],
[[project_fr13_tree_reshape_unifying_lever]], [[reference_scalar_metric_per_token_blindspot]],
[[feedback_check_artifact_before_concluding]], [[reference_diffuse_gdn_accumulation_explained]].
