# `release-note-to-plan-translation` Evaluator Contract

## Evaluation Goal
Score whether a solver can turn release notes into a concrete, dependency-aware implementation plan.

## Visible Checks
- `pytest -q tests/plan_contract.py`

## Trusted Checks
- Plan covers the benchmark-owned release intent.
- Output is structured and actionable.

## Hidden Checks
- Dependency correctness.
- Milestone sizing.
- Hidden risk surfaced from repo inventory.

## 100-Point Breakdown
- `10`: visible structure checks pass.
- `20`: plan correctness.
- `20`: dependency ordering.
- `20`: milestone sizing.
- `15`: risk handling.
- `15`: plan completeness.
