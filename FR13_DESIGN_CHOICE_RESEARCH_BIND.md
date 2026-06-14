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

**AND cat10 already ran the "wider" experiment (FR13_CAT10_BIND.md, verdict no_help):** root-sibling cat10
(10 nodes) = **22 clear-margin flips [2,6,8,6], FLAT vs cat9's 22 [6,6,4,6]** (redistributed p0 6->2,
p2 4->8, net zero) AND **accept/event 3.198 -> 2.932 (worse)** + ~+1.3% s/fwd. So **adding WIDTH does NOT
cut flips and HURTS accept** — the "shallower + wider root-sibling" reshape direction is empirically dead.
The open reshape lever is LEANER/shallower (chain5 / cat7 / cat8 subsets), which `chain5` is testing now.

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
