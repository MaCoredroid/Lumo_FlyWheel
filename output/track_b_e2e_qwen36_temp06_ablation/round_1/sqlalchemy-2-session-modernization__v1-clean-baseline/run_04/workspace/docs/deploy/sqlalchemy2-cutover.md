# SQLAlchemy 2 Cutover

## Changes Made

- **Repository layer** — replaced legacy `session.query()` with SQLAlchemy 2.x
  `select()` expressions using `session.scalars()` / `session.scalar()`.
- **Repository helpers no longer commit** — `create_entry` and other helpers are
  now pure mutation/read operations. Transaction ownership moves to the callers.
- **API** — `create_ledger_entry` and `read_ledger_status` no longer manage
  transactions themselves; callers use `session_factory.begin()` to create
  explicit transaction boundaries with automatic rollback on exceptions.
- **Worker** — `settle_entry` and `settle_batch` now open their own
  `session_factory.begin()` blocks. Each entry in a batch gets its own
  transaction so a failure rolls back only the failing entry and does not
  persist the intermediate "processing" status.
- **Admin CLI** — `plan_batch` and `reconcile_batch` use `session_factory.begin()`
  with per-entry transactions, ensuring failed operations are rolled back.

## Migration Notes

- No legacy compatibility shim is needed; all code paths now use SQLAlchemy 2.x
  semantics.
- All transaction boundaries are explicit at the API/worker/admin level, not
  hidden inside repository helpers.
