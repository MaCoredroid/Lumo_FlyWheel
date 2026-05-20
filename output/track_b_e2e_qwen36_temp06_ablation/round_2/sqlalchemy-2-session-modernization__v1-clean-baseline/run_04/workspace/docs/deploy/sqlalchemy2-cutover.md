# SQLAlchemy 2 Cutover

## What Changed

All repository queries migrated from legacy `session.query()` to SQLAlchemy 2.x
`select()` constructs. Transaction boundaries were moved out of the repository
layer and into the API, worker, and admin CLI layers.

## Repository (`app/repository.py`)

- `session.query()` replaced with `session.execute(select(...))` using
  `.scalar_one()` and `.scalar()` result helpers.
- `create_entry()` no longer calls `session.commit()`. The caller is now
  responsible for committing. This prevents premature transaction closure
  when the repository is used inside a `session_factory.begin()` block.

## API (`app/api.py`)

- `create_ledger_entry()` calls `session.flush()` so the row is visible to
  subsequent reads within the same transaction, but does not commit. The
  caller's `session_factory.begin()` context manager handles commit/rollback.
- `read_ledger_status()` is a plain read — no transaction boundary needed.

## Worker (`app/worker.py`)

- `settle_entry()` wraps the entire mark-and-settle sequence in a single
  transaction. On any exception the session is rolled back, so a failed
  settle never leaves a stale `"processing"` status.
- `settle_batch()` uses a per-entry transaction so one failure does not
  corrupt the batch session. Each entry is committed independently; on
  failure the current entry is rolled back and the exception propagates.

## Admin CLI (`app/admin_cli.py`)

- `plan_batch()` with `dry_run=True` now calls `session.rollback()` instead
  of `session.commit()`, so dry-run status changes are never persisted.
- `reconcile_batch()` uses per-entry transactions with the same dry-run
  rollback semantics.

## Rollback Behavior

| Component   | Commit point                     | Rollback on failure |
|-------------|----------------------------------|---------------------|
| Repository  | None (caller manages)            | N/A                 |
| API         | Caller's `begin()` context       | `begin()` context   |
| Worker      | After full settle succeeds       | Yes (try/except)    |
| Admin CLI   | After non-dry-run commit         | Yes (try/except)    |
