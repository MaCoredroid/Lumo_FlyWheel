# `multi-round-software-evolution` Evaluator Contract

## Evaluation Goal
Score whether a solver can plan the next round of software evolution with a sensible partial-progress metric.

## Visible Checks
- `pytest -q tests/test_evolution_plan.py`

## Trusted Checks
- Plan names one next-round focus and one explicit do-not-touch boundary.
- Partial-progress metric is concrete.

## Hidden Checks
- Long-horizon priority correctness.
- Slop-control via do-not-touch scope.
- Partial-progress metric quality.

## 100-Point Breakdown
- `10`: visible plan-shape checks pass.
- `20`: round-focus correctness.
- `20`: objective-delta quality.
- `20`: regression-free boundary setting.
- `15`: maintainability and slop control.
- `15`: partial-progress metric quality.
