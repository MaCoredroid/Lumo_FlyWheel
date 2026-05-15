# SQLAlchemy 2 Cutover

## Changes Made

1. **Repository layer**: Migrated from legacy `.query()` style to SQLAlchemy 2.x `select()` semantics. Repository functions no longer commit on their own; they return objects for the caller to manage transactions.

2. **API layer**: Transaction boundaries are now explicit at the API level. The caller uses `session_factory.begin()` context manager to manage transactions.

3. **Worker layer**: Added explicit try/except blocks with `session.rollback()` on failure to ensure status changes do not persist when exceptions occur.

4. **Admin CLI**: `dry_run=True` now properly rolls back changes instead of committing them.

## Transaction Boundaries

- **API**: Use `with session_factory.begin() as session:` for read/write operations
- **Worker**: Each operation has its own transaction with rollback on exception
- **Admin CLI**: Dry runs rollback changes; real runs commit

## Validation

Run the test suite to verify the modernization:

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```
