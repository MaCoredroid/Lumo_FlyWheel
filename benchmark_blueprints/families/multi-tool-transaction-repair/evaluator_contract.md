# `multi-tool-transaction-repair` Evaluator Contract

## Evaluation Goal
Score whether a solver can repair a stateful multi-service transaction without creating collateral damage.

## Visible Checks
- `pytest -q tests/test_transaction_flow.py tests/test_atomicity.py`

## Trusted Checks
- Transaction boundaries are repaired in code, not only docs.
- Atomicity guarantees remain intact.

## Hidden Checks
- Partial rollback correctness.
- Duplicate side-effect prevention.
- Correct final service state after retried execution.

## 100-Point Breakdown
- `10`: visible tests pass.
- `30`: final state correctness.
- `20`: atomicity and duplicate-side-effect guarantees.
- `20`: hidden partial-failure recovery.
- `10`: operator note accuracy.
- `10`: diff locality and bounded repair.

## Caps
- If duplicate side effects remain possible, total score is capped at `20`.
- If only docs are updated, total score is capped at `10`.
