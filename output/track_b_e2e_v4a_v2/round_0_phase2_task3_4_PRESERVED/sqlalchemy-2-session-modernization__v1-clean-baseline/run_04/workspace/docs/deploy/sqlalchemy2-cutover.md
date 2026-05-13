# SQLAlchemy 2 Cutover

This document describes the SQLAlchemy 2.x migration for the ledger-sync service.

## Changes Made

### Repository Layer
- Migrated from legacy `session.query()` to SQLAlchemy 2.x `select()` syntax
- Removed implicit `commit()` calls from repository functions
- Repository functions now operate within caller-managed transaction boundaries

### Transaction Boundaries
- **API layer**: Uses `session_factory.begin()` context for explicit transaction control
- **Worker layer**: Each operation uses its own transaction with rollback on failure
- **Admin CLI**: Dry-run mode uses `rollback()` to discard changes; normal mode commits

### Rollback Behavior
- All functions now properly rollback on exceptions
- Partial state changes are not persisted when errors occur
- Dry-run operations explicitly rollback instead of committing

## Validation

Run the test suite to verify the migration:

```bash
pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py
```
