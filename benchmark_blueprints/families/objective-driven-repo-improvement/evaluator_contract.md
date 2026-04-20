# `objective-driven-repo-improvement` Evaluator Contract

## Evaluation Goal
Score whether a solver can choose the best next intervention for a repo objective under governance and regression constraints.

## Visible Checks
- `pytest -q tests/test_objective_plan.py`

## Trusted Checks
- Output names one concrete next intervention.
- Governance constraints are obeyed.

## Hidden Checks
- Strategic choice correctness.
- Regression-risk reasoning.
- Expected-delta directionality.

## 100-Point Breakdown
- `10`: visible output checks pass.
- `20`: intervention choice.
- `20`: objective-delta quality.
- `20`: governance compliance.
- `15`: regression-risk analysis.
- `15`: bounded decision-note quality.
