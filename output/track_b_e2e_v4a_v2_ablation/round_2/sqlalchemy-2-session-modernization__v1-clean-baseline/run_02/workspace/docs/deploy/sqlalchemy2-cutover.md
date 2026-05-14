# SQLAlchemy 2 Cutover

## Summary

The ledger-sync service has been modernized to SQLAlchemy 2.x semantics with proper
transaction boundaries and rollback behavior.

## Changes

### Repository Layer
- Migrated from legacy `session.query()` to SQLAlchemy 2.x `select()` statements
- Removed automatic commits from repository helpers - callers now manage transactions

### API Layer
- Transaction boundaries are managed by the caller using `session.begin()` context
- Repository methods no longer commit; they return objects for the caller to persist

### Worker Layer
- Each operation has explicit transaction boundaries using `session_factory()` context
- Rollback on exception is automatic via context manager
- Batch operations use
per-item transactions to isolate failures

### Admin CLI Layer
- Dry run operations explicitly rollback changes before returning
- Transaction boundaries are explicit per operation
- Failures trigger automatic rollback via context manager

## Testing

Run the test suite to verify correct behavior:

```
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```

## Migration Notes

- Remove any legacy compatibility shims. Ensure all callers use explicit
transaction boundaries.
