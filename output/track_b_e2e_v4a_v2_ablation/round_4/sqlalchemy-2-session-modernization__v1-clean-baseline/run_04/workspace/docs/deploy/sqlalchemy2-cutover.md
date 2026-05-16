# SQLAlchemy 2 Cutover

## Summary

The `ledger-sync` service has been modernized to SQLAlchemy 2.x semantics with
explicit transaction boundaries and correct rollback behavior.

## Changes

### Repository Layer (`app/repository.py`)
- Migrated from legacy `session.query()` to SQLAlchemy 2.x `select()` style
- Removed implicit `commit()` calls from repository functions
- Repository functions now operate within caller-managed transactions

### API Layer (`app/api.py`)
- Uses `select()` for queries
- Does not manage transactions - caller controls commit/rollback via `session.begin()`

### Worker Layer (`app/worker.py`)
- Explicit transaction boundaries with try/except/rollback patterns
- All mutations within a single transaction; partial commits removed
- Failures trigger rollback to maintain data consistency

### Admin CLI (`app/admin_cli.py`)
- `dry_run=True` now uses `rollback()` instead of `commit()` to prevent persistence
- Proper exception handling with rollback on failure

## Testing

Run validation tests:
```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```

## Migration Notes

- No legacy compatibility shim required
- Session factory uses `future=True` for 2.x semantics
- Transaction management is now explicit at the business logic layer
