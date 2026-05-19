# SQLAlchemy 2 Cutover

Repository access now uses SQLAlchemy 2-style `select()` statements instead of the
legacy `session.query()` API. The repository no longer commits on its own —
transaction boundaries are owned by the caller.

API, worker, and admin CLI each manage their own explicit transaction
boundaries with correct rollback behavior:

- **API** — the caller (framework or test) opens a `session_factory.begin()`
  block; repository helpers stay within that transaction without committing.

- **Worker** — `settle_entry` and `settle_batch` use `session_factory.begin()`
  so the entire settle operation is atomic. On failure the session rolls back
  automatically, leaving the entry in its previous state.

- **Admin CLI** — `plan_batch` rolls back on `dry_run=True` instead of
  committing. `reconcile_batch` uses `session_factory.begin()` with automatic
  rollback on failure.
