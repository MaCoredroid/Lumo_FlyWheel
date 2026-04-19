# `sqlalchemy-2-session-modernization` Evaluator Contract

## Evaluation Goal
Score whether a solver performs a real SQLAlchemy 2.0 transaction-boundary modernization across API, worker, and admin CLI surfaces.

## Visible Checks
- `pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py`

## Trusted Checks
- No live `session.query(` remains on the service path.
- No global-session reuse or blanket helper-level commit was introduced.
- Deploy note names the transaction-boundary change and rollback steps.

## Hidden Checks
- Worker retry path is idempotent after forced failure.
- Dry-run mode performs no writes.
- Rollback leaves database state clean after exceptions.

## 100-Point Breakdown
- `10`: visible tests all pass.
- `5`: `session.query(` is removed from live paths.
- `5`: deploy note is specific and operationally correct.
- `20`: hidden worker retry pack proves rollback and idempotence.
- `20`: hidden admin CLI pack proves dry-run non-persistence and batch correctness on withheld fixtures.
- `20`: hidden rollback-state checks pass across forced-failure cases.
- `10`: trusted anti-global-session and anti-blanket-commit checks pass.
- `10`: hidden multi-entrypoint transaction-boundary checks pass.

## Caps
- If the worker path still leaks partial writes, total score is capped at `25`.
- If the solver introduces blanket commits or global session reuse, total score is capped at `20`.
- If only the visible entrypoints are proven and no withheld retry/idempotence pack is exercised, total score is capped at `20`.
- If the modernization is only syntactic and does not prove behavior across entrypoints, total score is capped at `20`.
- If tests are bypassed or removed, total score is capped at `0`.

## Passing Interpretation
- Target benchmark-design outcome: a naive GPT-5.4/high solver should land around `20/100`.
- Scores above `30` indicate the migration family still over-rewards mechanical rewrites.
