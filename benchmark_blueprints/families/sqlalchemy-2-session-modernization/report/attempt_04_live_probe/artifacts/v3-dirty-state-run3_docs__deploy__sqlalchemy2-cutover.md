# SQLAlchemy 2 Cutover

This service now uses SQLAlchemy 2-style repository access and caller-owned
transaction boundaries.

- Repository helpers use `select(...)`/`Session.execute(...)` and never commit.
- API request handlers are expected to run inside an explicit session boundary
  such as `with session_factory.begin() as session: ...`.
- Worker flows wrap each settle operation in a single `session_factory.begin()`
  block so failures roll back intermediate status changes.
- Admin CLI flows use the same explicit boundary. Dry-run paths intentionally
  roll back before returning, and batch failures roll back the full batch.

Do not reintroduce helper-level commits or a compatibility shim during the
cutover. The service relies on outer transaction scopes to make rollback
behavior predictable across the API, worker, and admin surfaces.
