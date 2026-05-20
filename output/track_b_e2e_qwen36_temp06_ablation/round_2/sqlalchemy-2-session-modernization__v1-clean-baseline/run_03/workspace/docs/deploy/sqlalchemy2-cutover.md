# SQLAlchemy 2 Cutover

Repository helpers now use SQLAlchemy 2-style `select()` statements instead of
the legacy `session.query()` API. The internal `commit()` calls have been
removed from repository functions so that callers own the transaction boundary.

The API, worker, and admin CLI modules each manage explicit transaction
boundaries:

- **API** — callers wrap operations in `session_factory.begin()`; repository
  functions no longer commit, so the caller's transaction stays active.
- **Worker** — `settle_entry` and `settle_batch` perform all status changes
  within a single transaction and commit only on success. If an exception is
  raised the session context manager rolls back automatically, leaving the
  original status intact.
- **Admin CLI** — `plan_batch` rolls back when `dry_run=True` so no status
  change persists. `reconcile_batch` commits only when `dry_run=False` and
  rolls back on failure.
