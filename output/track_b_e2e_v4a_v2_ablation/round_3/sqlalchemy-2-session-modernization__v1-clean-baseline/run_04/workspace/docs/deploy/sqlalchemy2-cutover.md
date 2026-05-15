# SQLAlchemy 2 Cutover

## Summary

The ledger-sync service has been modernized to use SQLAlchemy 2.x semantics with proper transaction boundaries.

## Changes

### Repository Layer

- Replaced legacy `session.query()` with SQLAlchemy 2.x `select()` statements`
- Removed implicit commits from repository helpers
- All data access now uses explicit `session.execute()` with `scalar()` or `scalars()`

### Transaction Boundaries

- **API**: Transaction boundaries are managed by caller; repository helpers no longer commit
- **Worker**: Single transaction per operation with `flush()` for intermediate state; rollback on exception
- **Admin CLI**: Explicit rollback for dry-run mode; commit only for actual operations

## Rollback Behavior

- `settle_entry`: On exception after marking "processing", transaction rolls back to "pending"
- `settle_batch`: On exception, entire batch rolls back
- `plan_batch` (dry_run=True): Status changes are rolled back, no persistence
- `reconcile_batch` (dry_run=True): All status changes are rolled back at end
