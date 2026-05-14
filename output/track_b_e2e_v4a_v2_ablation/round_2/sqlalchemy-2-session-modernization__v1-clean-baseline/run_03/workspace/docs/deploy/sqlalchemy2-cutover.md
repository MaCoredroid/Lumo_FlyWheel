# SQLAlchemy 2 Cutover

## Summary

This cutover migrates the ledger-sync service to SQLAlchemy 2.x semantics with explicit
transaction boundaries. Repository functions no longer manage commits internally;
instead, callers control transaction scope with proper rollback behavior.

## Changes

### Repository Layer (`app/repository.py`)

- Converted from legacy `.query()` API to SQLAlchemy 2.x `select()` statements
- Removed internal `session.commit()` calls from repository functions
- Functions now return results without persisting; callers manage transactions

### API Layer (`app/api.py`)

- Updated to use `select()` statements directly
- Transaction boundaries managed by caller (e.g., `with session_factory.begin()`)

### Worker Layer (`app/worker.py`)

- Uses `session.flush()` to mark changes without committing
- Only commits after all operations succeed
- Session context manager automatically rolls back on exceptions

### Admin CLI (`app/admin_cli.py`)

- `dry_run=True` now uses `session.rollback()` to discard changes
- Explicit `try/except` with `session.rollback()` on failure

## Testing

Run the test suite to verify transaction boundary behavior:

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```
