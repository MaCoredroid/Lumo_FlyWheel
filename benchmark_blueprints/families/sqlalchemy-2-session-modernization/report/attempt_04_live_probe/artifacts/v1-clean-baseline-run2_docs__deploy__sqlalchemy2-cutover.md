# SQLAlchemy 2 Cutover

The SQLAlchemy 2 cutover removes legacy `session.query(...)` usage from the
repository layer and moves transaction ownership to the application boundary.

- Repository helpers use SQLAlchemy 2-style `select(...)` statements and never
  call `commit()` on behalf of their callers.
- API request handlers should open an explicit transaction, typically with
  `session_factory.begin()`, and then call repository helpers within that unit
  of work.
- Worker jobs should treat each settle operation or batch as a single
  transaction so failures roll back intermediate status changes instead of
  persisting `processing`.
- Admin CLI commands should use explicit transactions for live runs and explicit
  rollbacks for `dry_run` planning so queued status updates are never persisted
  accidentally.
