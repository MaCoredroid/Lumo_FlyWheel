# SQLAlchemy 2 Cutover

The cutover moves transaction ownership out of repository helpers and into the
API, worker, and admin entry points.

- Repository access uses SQLAlchemy 2-style `select(...)` statements and never
  commits on behalf of callers.
- API handlers should open an explicit unit of work with
  `session_factory.begin()` so reads and writes happen inside one transaction.
- Worker operations run each settle flow inside a single transaction and roll
  back the whole unit of work if an exception is raised before commit.
- Admin commands keep dry-run changes transient by flushing for validation and
  then rolling back instead of committing queued status updates.

This preserves behavioral safety during the SQLAlchemy 2 migration: repository
helpers stay side-effect free, and failure paths no longer leak intermediate
`processing` or `queued` states into the database.
