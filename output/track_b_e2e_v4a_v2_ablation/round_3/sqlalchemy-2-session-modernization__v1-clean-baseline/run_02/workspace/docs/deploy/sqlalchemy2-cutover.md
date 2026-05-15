# SQLAlchemy 2 Cutover

All repository functions now use SQLAlchemy 2.x `select()` syntax instead of
legacy `session.query()`. Transaction boundaries are explicit in the API,
worker, and admin CLI layers:

- **API layer**: Functions no longer commit internally; callers manage
  transaction boundaries using `session.begin()` or explicit commit/rollback.

- **Worker layer**: Each operation wraps work in try/except with rollback on failure.
  The `settle_entry` and `settle_batch` functions commit after marking
  "processing", then rollback if an exception occurs before final commit.

- **Admin CLI layer**: The `plan_batch` function rolls back on dry_run to avoid
  persisting status changes. The `reconcile_batch` function also rolls back
  on dry_run and on any exception.

Migration notes:
- Remove any legacy compatibility shims that were previously in place.
- Ensure all callers use `session_factory.begin()` context when they need
  transactional guarantees.
- Repository helpers no longer call `session.commit()` internally.
