# SQLAlchemy 2 Cutover

Repository access now uses SQLAlchemy 2-style `select()` statements instead of
the legacy `session.query()` API. The `session.commit()` call was removed from
repository helpers so that transaction ownership lives at the call site.

The API, worker, and admin CLI each manage explicit transaction boundaries:

- **API** — callers pass a session; the outer `session_factory.begin()` context
  controls commit / rollback.
- **Worker** — `settle_entry` and `settle_batch` commit once at the end of the
  operation. If an exception is raised before `commit()`, the session context
  manager rolls back, so intermediate status changes (e.g. "processing") never
  persist.
- **Admin CLI** — `plan_batch` and `reconcile_batch` call `session.rollback()`
  in dry-run mode so no status mutations are written. Non-dry-run paths commit
  after all entries are processed.
