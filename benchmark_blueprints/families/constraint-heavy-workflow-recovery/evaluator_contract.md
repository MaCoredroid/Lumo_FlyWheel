# `constraint-heavy-workflow-recovery` Evaluator Contract

## Evaluation Goal
Score whether a solver can recover a partial workflow without breaking one constraint to satisfy another.

## Visible Checks
- `pytest -q tests/test_constraints.py tests/test_recovery.py`

## Trusted Checks
- Code, not only docs, enforces the recovery policy.
- Recovery boundaries are explicit and accurate.

## Hidden Checks
- Multi-constraint combinations on withheld partial-run fixtures.
- Correct handling of irreversible prior steps.
- No silent loss of required reporting.

## 100-Point Breakdown
- `10`: visible tests pass.
- `30`: final state correctness.
- `25`: policy and constraint obedience.
- `20`: hidden recovery-path correctness.
- `15`: recovery note accuracy.

## Caps
- If any core constraint is violated, total score is capped at `20`.
- If irreversible-step replay is wrong, total score is capped at `10`.
