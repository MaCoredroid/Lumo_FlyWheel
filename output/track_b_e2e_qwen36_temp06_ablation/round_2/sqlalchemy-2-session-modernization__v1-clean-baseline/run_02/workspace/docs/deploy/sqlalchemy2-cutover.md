# SQLAlchemy 2 Cutover

## Summary

Migrated repository queries to SQLAlchemy 2.x `select()` semantics and
introduced explicit transaction boundaries in the API, worker, and admin CLI.

## Changes

### Repository (`app/repository.py`)

- Replaced legacy `session.query()` calls with `session.execute(select(...))`.
- Removed internal `session.commit()` from `create_entry` — callers now own
  the transaction boundary.

### API (`app/api.py`)

- No code changes required; the API now relies on the caller-provided
  transaction (e.g. `session_factory.begin()`) for both create and read
  operations.

### Worker (`app/worker.py`)

- `settle_entry` and `settle_batch` now wrap all mutations in a single
  `session_factory.begin()` block.  Intermediate commits were removed so that
  a failure at any point rolls back the entire operation — the "processing"
  status is no longer persisted on error.

### Admin CLI (`app/admin_cli.py`)

- `plan_batch` and `reconcile_batch` now use `session_factory.begin()`.
- Dry-run paths call `session.rollback()` instead of `session.commit()`, so
  status changes are never persisted during a dry run.

## Validation

Run the test suite to confirm correctness:

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```
