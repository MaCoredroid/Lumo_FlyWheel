# `backlog-decomposition-scheduler` Evaluator Contract

## Evaluation Goal
Score whether a solver can turn a backlog into a dependency-aware, capacity-aware execution schedule.

## Visible Checks
- `pytest -q tests/test_schedule_contract.py`

## Trusted Checks
- Output is structured and internally consistent.
- Schedule advances the stated objective.

## Hidden Checks
- Capacity and specialization constraints.
- Dependency correctness.
- Risk-isolation sequencing.

## 100-Point Breakdown
- `10`: visible schedule-shape checks pass.
- `20`: schedule correctness.
- `20`: dependency handling.
- `20`: capacity handling.
- `15`: risk isolation.
- `15`: objective progress quality.
