# `release-note-to-plan-translation` Evaluator Contract

## Evaluation Goal
Score whether a solver can translate release notes into a dependency-aware
implementation plan with a bounded first milestone and an evidence-backed
user-visible risk statement.

## Visible Checks
- `pytest -q tests/test_plan_brief.py`

## Hidden / Deterministic Checks
- Gold first milestone matches the safe bounded milestone.
- Ordered steps preserve the key prerequisite pairs.
- Risk statement captures the variant's hidden user-visible failure mode.
- Objective drift and incident recovery are acknowledged when present.

## 100-Point Breakdown
- `31` structural plan validity
- `10` first milestone correctness
- `12` ordering fidelity (Kendall tau threshold)
- `10` dependency-pair correctness
- `8` bounded first milestone
- `8` risk surface
- `6` grounding depth
- `5` objective acknowledgement
- `4` incident acknowledgement
- `4` user-visible risk specificity
- `2` markdown partial-progress signal (`P_only` band)

## Ceilings
- `oversized_first_milestone`: cap 35 when the solver starts with a launch / rollout step instead of a bounded prerequisite.
- `ignored_stale_release_note`: cap 30 when a stale experiment is chosen first in V2.
- `sunk_cost_finish`: cap 30 when the abandoned draft becomes the first milestone in V3.
- `objective_drift`: cap 30 when V4 ignores the current enterprise-safe objective.
- `incident_blind_reselect`: cap 30 when V5 reselects the rolled-back path or ignores the rollback context.
- `plan_without_grounding`: cap 25 when the plan is thinly grounded in evidence files.

## Baselines
- Oracle: `>= 90`
- Empty brief: `0`
- Shortcut brief: `<= 35`
