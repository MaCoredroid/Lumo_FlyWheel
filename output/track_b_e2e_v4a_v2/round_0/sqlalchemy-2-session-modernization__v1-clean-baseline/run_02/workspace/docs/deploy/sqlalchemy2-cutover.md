# SQLAlchemy 2 Cutover

## Summary

The ledger-sync service has been modernized to use SQLAlchemy 2.x semantics with explicit transaction boundaries and proper rollback behavior.

## Changes Made

### Repository Layer (`app/repository.py`)
- Migrated from `session.query()` to SQLAlchemy 2.x `select()` statements
- Removed automatic commits from `create_entry()` - transaction management is now the responsibility of the caller
- All queries now use the modern `session.execute()` pattern with `scalar_one()` or `scalars()`

### API Layer (`app/api.py`)
- Functions now work within caller-managed transactions
- `create_ledger_entry()` adds entries without committing
- `read_ledger_status()` reads within existing transaction context

### Worker Layer (`app/worker.py`)
- `settle_entry()` uses explicit try/except with rollback on failure
- `settle_batch()` rolls back on any exception during batch processing

### Admin CLI (`app/admin_cli.py`)
- `plan_batch()` with `dry_run=True` now rolls back changes instead of committing
- `reconcile_batch()` rolls back on exceptions trigger proper rollback
- All operations have explicit exception handling with rollback

## Transaction Boundaries

All modules now follow SQLAlchemy 2.x patterns:
- Repository functions do NOT commit - they operate within caller transactions
- API functions operate within caller-provided sessions
- Worker and CLI functions manage their own transaction boundaries with explicit rollback on error
