# SQLAlchemy 2 Cutover

Use SQLAlchemy 2-style ORM access during the cutover.

- Repository reads should use `select(...)` execution patterns instead of the
  legacy `session.query(...)` API.
- Repository helpers must not call `commit()` on behalf of callers. They may
  `add()` objects and `flush()` when the caller needs generated state inside the
  current transaction.
- The API request scope, worker jobs, and admin CLI commands each own their
  transaction boundary explicitly. Successful units of work commit once at the
  top level; failures roll back the entire unit of work.
- Dry-run admin flows should still exercise ORM writes inside the transaction,
  then end with an explicit rollback so no status change persists.
