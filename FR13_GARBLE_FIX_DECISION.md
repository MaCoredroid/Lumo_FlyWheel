# FR13 garble — fix decision (both targeted fixes failed → accept within-floor)

**Date:** 2026-07-10. Closes the garble-fix drive with an evidence-based decision after
proving the mechanism and testing both candidate fixes head-to-head.

## What is solid (proven this session)

- **Mechanism MEASURED** (FR13_GARBLE_DRIFT_BINDING_PROVEN.md): the misspell is tree-verify
  FORWARD DRIFT. no-spec masks the near-neighbor at ~1e-6; the tree-verify forward inflates it
  into the top-p nucleus (committed_prob measured 0.08 at the actual commit node via the
  commit-trace instrument); the PROVEN-correct committer (commits at exactly target_prob_draft,
  offline gate 22/22) faithfully emits it. 13/16 localized garbles are wrong-accepts.
- **The committer is NOT the bug** — a genuinely-1e-6 token cannot commit at ~8% through a
  committer proven to commit at target_prob_draft. (Refutes the stale 36f645d0 "accept-logic"
  claim, banner-corrected in FR13_DRIFT_FIX_PLAN.md.)

## Fix candidates — both tested, both failed

**Track 1 — fp32 tree-conv seed fix: NO-GO** (premise-check wf_2233d7b7, over-determined):
- The drift is NOT L0-conv-seed-dominated. Ladder arithmetic (output/fr13_node5_ladder):
  fp32-conv cuts drift **≤2x** vs the **5.71x** (=10.7/1.875) needed. The dominant amplifier is
  **full-attn (1.24x/layer)**, which conv-fp32 doesn't touch; a valid localization puts a flip's
  birth at **L56**, not L0; the deep-accept garble's L0 source is the **GDN scan**, not the conv.
- Conv is already bit-exact to native (reduction-order diff, not accuracy); SSM `h` is ALREADY
  fp32 yet the garble persists; fp32-tree is asymmetric vs the bf16 no-spec reference.
- Validating first SAVED a multi-hour dozens-of-dtype-sites refactor of a change the data
  predicts would be inert (or worse). Reverted (parked note at patcher :2543; launcher
  MAMBA_CACHE_DTYPE knob retained, default off).

**Track 2 — committer drift-band bias: INEFFECTIVE** (live A/B, flag LUMO_TREE_COMMIT_DRIFT_BIAS):
- Design: at confident nodes (tree argmax ≥ τ_high 0.85) zero the low-band tail (p < τ_low 0.20,
  keep argmax) + renorm. Default OFF byte-identical (offline gate OFF=22/22). Unit-validated.
- Live A/B: the bias ENGAGED (0 confident-band non-argmax commits vs 23 baseline) YET the
  localizer still found **18 WRONG_ACCEPT** (baseline 13 — no reduction). The garbles commit at
  **argmax_prob ~0.80 (< τ_high)** — the drift is strong enough to depress the correct token's
  argmax to ~0.80, so the garbles sit right at the threshold and the bias misses them. It is
  **boot-fragile** (baseline garble at argmax 0.92 = catchable; this boot 0.80 = not) and
  **blunt** (garbles overlap genuine commits in (argmax,committed_prob) space — no clean
  threshold). Kept default-OFF as a tunable diagnostic; NOT recommended (does not work).

## COMMITTER-SIDE MITIGATION ABANDONED AS A CLASS (user decision 2026-07-11)

Not just the one bias — the ENTIRE committer-side direction is abandoned, on a first-principles
wall: the committer only ever sees the DRIFTED target. It commits each token at exactly its
target prob (SpecInfer canonical multi-draft, symmetric across a node's children; no spine-first
priority; proven, offline gate 22/22). The garble is a two-candidate node (correct spine token +
near-neighbor branch) where the committer commits the branch at the drifted target's inflated prob.
Any committer fix must either (a) stay distribution-correct → commits the same drifted prob → no
help, or (b) bias spine>branch → not lossless AND ineffective (the drift depresses the correct
argmax to ~0.80 so garbles slip any threshold; blunt; boot-fragile). The committer cannot tell a
drift-inflated garble (true ~1e-6) from a genuine alternative (true ~0.2) without the no-spec
target, which spec-decode does not compute. => no committer-side lever can be both clean and
effective. The drift-band bias code was DELETED (commit-trace instrument kept); the committer is
distribution-lossless again.

## Decision: ACCEPT within-floor; reshape is the on-demand escape hatch

Both cheap/targeted fixes failed on evidence; the forward-drift fix is exhausted (M-invariance
compute-only + fp32-conv both NO-GO) and the committer fix is ineffective (the drift is diffuse
and can dominate the argmax, not a boundary-crosser a threshold catches). The garble is:
- already WITHIN the accepted lossless floor (committer proven correct; intermittent; low-margin;
  correct spelling dominates ~51/59 live);
- fixable ONLY by tree-reshape (chain5, removes co-resident branches → co-residency drift
  structurally absent → the only proven kill) at a MEASURED **~7-9% throughput** cost.

Per "speed is the goal, correctness a within-floor constraint": **keep cat8 (full tree speed),
document the garble as a within-floor forward-drift artifact, and hold reshape as a runtime
escape hatch** if the garble ever measurably bites a real SWE-Verified deliverable. The mechanism
+ full fix analysis are banked; nothing here is a dead-end left unexamined (3 adversarial
workflows + a binding measurement + a live A/B).
