# SQLAlchemy 2 Cutover

## Changes Made

### Repository — SQLAlchemy 2.x selects

All legacy `session.query()` calls in `app/repository.py` have been replaced
with `session.execute(select(...))` using SQLAlchemy 2.x `select()` constructs.

- `get_entry` — uses `select(LedgerEntry).where(...)` with `.scalar_one()`
- `entry_exists` — uses `select(LedgerEntry.id).where(...)` with `.scalar_one_or_none()`
- `pending_entry_count` — uses `select(LedgerEntry).where(...)` with `.scalars(...).count()`
- `create_entry` — no longer commits; the caller owns the transaction boundary

### Transaction boundaries

Each component now manages its own transaction scope with explicit rollback:

- **API** (`app/api.py`) — callers pass a session created via
  `session_factory.begin()`; the repository no longer commits inside
  `create_entry`, so the outer context manager controls commit/rollback.
- **Worker** (`app/worker.py`) — `settle_entry` and `settle_batch` wrap
  mutations in `try/except` blocks that call `session.rollback()` before
  re-raising, so a failure never leaves a stale "processing" status.
- **Admin CLI** (`app/admin_cli.py`) — `plan_batch` rolls back on dry-run
  instead of committing. `reconcile_batch` rolls back on failure or on
  dry-run, preserving the original data.

## Rollback plan

Revert to the previous git commit if any integration test fails against
the production database.
