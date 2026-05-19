# SQLAlchemy 2 Cutover

Repository helpers now use SQLAlchemy 2-style `select()` queries instead of the
legacy `session.query()` API. Repository functions no longer commit — they
delegate transaction boundaries to the caller.

The API, worker, and admin CLI each manage their own explicit transaction
boundaries with proper rollback on failure:

- **API** — callers use `session_factory.begin()` to wrap reads and writes in a
  single transaction.
- **Worker** — `settle_entry` rolls back the entire transaction when an error
  occurs after marking an entry "processing", so the status reverts to its
  previous value.
- **Admin CLI** — `plan_batch` with `dry_run=True` rolls back instead of
  committing, so no status change persists.
