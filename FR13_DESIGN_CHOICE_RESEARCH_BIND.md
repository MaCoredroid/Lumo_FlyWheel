# FR13 — diffuse design-choice research: geometry seam RE-REFUTED, no clean small fix; decisive control = chain5

Date 2026-06-14. CPU workflow `wrzufs44h` (`wf_0d7caa9b-bc8`), Mechanism + DesignChoices + Verify.
**Verify holds=FALSE** — it caught the Mechanism's central error. Raw:
`research/fr13_workflows/diffuse_design_research_wrzufs44h.raw.json`.

## The Mechanism phase's "launch-geometry seam" is MEASURED-FALSE (re-refuted)
The Mechanism agent concluded the cat9-vs-native excess = a GDN-scan launch-geometry (BV=16/warps=8 vs
native BV=32/warps=4) reduction-order seam, ~1 ULP/node compounding. **The Verify refuted it on our own
banked measurement:** the BV A/B (`woaiybls4`, `FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND`, verify holds=True)
found RAW max_abs == 0.0 at **all four cells incl. D32_npad16** (the deployed deep regime) vs the genuine
native kernel — our scan already matches native at both geometries; there is NO reduction-order gap to
close. Also internally inconsistent: the flip crystallizes at L60/L61 (diffuse tail), not at the scan
output. So the BV/spill/recompute line stays SHELVED. (A stale-trail read: the Mechanism cited the 01:16
seam-scan hypothesis but not the 02:25 refutation.)

## No clean small lossless+speed-neutral design choice found
- Candidate A (recompute spine state at native BV=32/w4): MOOT (fixes the absent geometry seam) + spills
  636 B/thread at N_PAD=16.
- Candidate C (confidence-gated accept depth): correctly rejected (deviates the accept distribution =
  lossless risk).
- Tree-reshape: complementary, being GPU-tested — see the drafter constraint below.

## CORRECTION (2026-06-14, user caught it): the "no root sibling" drafter constraint is FALSE
The sweep design agent claimed the drafter is hardcoded to spine + depth-2-5 top-2 leaf, "no root sibling
`(1,)`, no rank>=2." **This is WRONG — cat10 disproves it.** cat10's TREE is
`[(0,), (1,), (0,0), (1,0), (0,0,0), (1,0,0), (0,0,0,0), (1,0,0,0), (0,0,0,0,0), (1,0,0,0,0)]` — a root
sibling `(1,)` PLUS a full second 5-chain off it. The MTP drafter produces the rank-1 root token fine;
building a wider tree is just a different `TREE` env (num_spec auto-derived). So the realizable reshape
space is NOT limited to subsets of cat9; root-sibling / wider trees ARE buildable with no retrain.
**rank>=2 (top-3+) feasibility is still open** (re-check against the code, not the agent's claim).

**SECOND CORRECTION (2026-06-14, user caught it again): "wider hurts accept" is ALSO WRONG.** My read of
cat10 (accept 3.198->2.932 = "wider hurts") was hand-wavey and CONTRADICTS the standing directive +
FR13_CAT10_INVESTIGATE_BIND.md (`wf_59bf2440`, verify holds=FALSE corrected it). The truth:
- The accept drop is **NOT apples-to-apple**: cat9 and cat10 run DIFFERENT greedy streams (diverge early;
  cat10 p0 hit EOS 25 tokens sooner = 73 vs 98 over 1 MORE event) = class-12 whole-window denominator
  confound. Not a controlled comparison.
- It is **NOT leaf-lossiness**: m1 (verify co-residency) STRUCTURALLY RULED OUT (strict_mask makes the
  root-sibling row attention-invisible to every spine row — cannot contaminate spine acceptance); m3
  (commit handoff) inert (sibling commits accepted_len=1). The d0->d1 drop is the **sibling-stop
  denominator artifact** (sibling win caps at d0, deflating d1|d0; de-confounded d1|d0 recovers ~0.84+,
  d2-d4 FLAT). Any residual real dilution is sub-ULP (extra row in the 16-pad tile) and UNMEASURED.
- The root branch's **d0-rescue is REAL** (27% on near-tie roots); the future lever is the
  CONFIDENCE-GATED root branch (FREE top2-margin gate). So root-sibling/wider is NOT "dead."

What IS defensible: cat10 (full root branch) does **not REDUCE the 22-flip** — cat9 [6,6,4,6] and cat10
[2,6,8,6] are 100% DISJOINT positions, same total 22, because the 22-flip is a **per-forward channel-2
GDN-diffuse defect ORTHOGONAL to tree topology** (the same defect repositioned by the trajectory, NOT
caused by width). So reshape REPOSITIONS the 22, and whether ANY topology change can REDUCE it is exactly
what `chain5` tests (chain5~3 => topology/branches matter; chain5~22 => it's the per-forward kernel/handoff,
topology-independent => reshape is a dead end for the flip count). Lesson: do NOT take cross-trajectory
accept/event deltas at face value (class-12 confound); use per-node de-confounded counters.

## Live suspects (Verify) + the DECISIVE control
With geometry dead, the remaining sources of the cat9(22)-vs-native(3) excess are: (i) **tree co-residency**
(cat9's 9-node tree forward carries 4 branch rows alongside the spine; native's chain forward has none) and
(ii) **cross-event h0 handoff** (ours fp32 multi-column bank vs native bf16 single-slot roll). Both are
about the TREE/multi-row forward, not a single op.

**Decisive experiment = `chain5`** (`[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]`, pure 5-spine = native's
topology through OUR tree kernel), gated by the same full-stream flip-count probe:
- chain5 ~3 flips => the LEAVES (co-residency) are the excess => lean the tree (cat7/cat8 subsets) maps the
  flips-vs-accept frontier (note chain5 itself gives up cat9's accept edge = diagnostic, not deployable).
- chain5 ~22 flips => our kernel/handoff on a pure chain differs from native even WITHOUT branches => the
  excess is the h0-handoff / forward-shape, not the branches => reshape can't fix it; align the chain path.

NOTE: the 22 were measured vs the no-spec DECODE oracle (the deployment-correct lossless ground truth, same
as native's 3). chain5 keeps that same gate. Run via the driver `scripts/fr13_shape_gate.sh` DIRECTLY (the
sweep workflow failed because the ~512-request teacher-force outran the agent's StructuredOutput nudge
budget; the driver is deterministic + has a teardown trap). Pairs with
[[reference_diffuse_gdn_accumulation_explained]], [[project_fr13_tree_reshape_unifying_lever]],
[[feedback_no_reroute_reward_hacking]], [[feedback_kill_wrong_gpu_task_immediately]].
