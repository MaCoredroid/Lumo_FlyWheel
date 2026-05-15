# SQLAlchemy 2 Cutover

## Summary

This cutover modernizes the ledger-sync service to SQLAlchemy 2.x semantics with
proper transaction boundaries across the API, worker, and admin CLI layers.

## Changes

### Repository Layer

- Converted all queries from legacy `session.query()` to SQLAlchemy 2.x `select()`
- Removed internal `session.commit()` calls from repository functions
- Repository functions now only prepare statements and modify objects;
  transaction management is handled by callers

### API Layer

- API functions rely on explicit transaction boundaries managed by the caller
- Uses `session.begin()` context for atomic operations

### Worker Layer

- Each operation has explicit transaction boundary with `session_factory()` context
- On exception, the context manager automatically rolls back changes
- Commit only occurs on successful completion

### Admin CLI Layer

- Dry run operations explicitly call `session.rollback()` to
- prevent persistence
- Normal operations commit only after all validations pass

## Testing

Run the test suite to verify transaction behavior:

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```
