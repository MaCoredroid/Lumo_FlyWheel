# SQLAlchemy 2 Cutover

This document describes the SQLAlchemy 2.x migration for the ledger-sync service.

## Changes Made

### Repository Layer
- Migrated from legacy `.query()` syntax to SQLAlchemy 2.x `select()` statements
- Removed internal `session.commit()` calls from repository methods
- Repository methods now operate within caller-managed transaction boundaries

### Transaction Boundaries
- **API layer**: Uses `session.begin()` context for explicit transaction control
- **Worker layer**: Wrappers**: Wraps operations in try/except with explicit rollback on failure
- **Admin CLI**: Wraps operations in try/except with explicit rollback on failure
- Dry-run operations rollback changes before returning

## Key Behaviors

### On Failure
- All failed operations trigger `session.rollback()` to undo partial changes
- Exceptions propagate to callers for handling

### Dry-Run Mode
- Admin CLI dry-run operations explicitly rollback after reading data
- No persistent changes are made during dry-run

## Verification

Run the test suite to validate the migration:

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```
