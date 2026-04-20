# `partial-failure-rollback-recovery` Evaluator Contract

## Evaluation Goal
Score whether a solver can restore local consistency after a failed maintenance action without destroying audit or retry evidence.

## Visible Checks
- `pytest -q tests/test_rollback.py tests/test_ledger.py`

## Trusted Checks
- Rollback logic and ledger logic agree.
- Ops note reflects retryable versus terminal failure handling.

## Hidden Checks
- Consistent state after withheld failed-run fixtures.
- Preservation of audit or retry evidence.
- Correct handling of retryable failures.

## 100-Point Breakdown
- `10`: visible tests pass.
- `30`: final state correctness.
- `20`: evidence preservation.
- `20`: hidden rollback-path correctness.
- `10`: retryable-versus-terminal classification.
- `10`: ops note quality.

## Caps
- If audit evidence is deleted, total score is capped at `10`.
- If state remains inconsistent after rollback, total score is capped at `20`.
