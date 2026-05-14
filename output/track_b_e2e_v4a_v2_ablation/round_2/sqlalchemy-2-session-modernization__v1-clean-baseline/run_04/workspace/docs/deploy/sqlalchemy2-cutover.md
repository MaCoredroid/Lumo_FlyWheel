# SQLAlchemy 2 Cutover

## Changes Made

1. **Repository layer**: Migrated from legacy `session.query()` to SQLAlchemy 2.x `select()` syntax.
   - `get_entry()` now uses `select(LedgerEntry).where(...))` with `session.execute()`
   - `entry_exists()` uses `select(LedgerEntry.id)` with `session.execute().scalar()`
   - `pending_entry_count()` uses `select(LedgerEntry)` with `session.execute().scalars().count()`
   - `create_entry()` no longer commits - transaction handled by caller

2. **Worker layer**: Explicit transaction boundaries with rollback on error.
   - `settle_entry()`: Uses `session.flush()` to mark changes, then `rollback()` on failure before `commit()` on success
   - `settle_batch()`: Commits once after all entries processed, rolls back on any failure

3. **Admin CLI layer**: Proper dry_run semantics.
   - `plan_batch()`: Uses `rollback()` for dry_run to prevent status changes from persisting
   - `reconcile_batch()`: Uses `rollback()` for dry_run or on failure, `commit()` only for actual execution

## Transaction Semantics

- API layer: Caller manages transaction boundary via `session_factory.begin()`
- Worker layer: Functions manage their own transaction with explicit rollback on error
- Admin CLI layer: Functions manage their own transaction with rollback for dry_run or on error
