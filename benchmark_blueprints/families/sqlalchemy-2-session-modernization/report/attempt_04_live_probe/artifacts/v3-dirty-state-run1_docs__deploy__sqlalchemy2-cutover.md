# SQLAlchemy 2 Cutover

This cutover removes legacy `session.query(...)` usage from repository access
and moves transaction ownership to the service entrypoints.

- Repository helpers must use SQLAlchemy 2.x `select(...)` patterns and must not
  call `commit()` themselves.
- API request handlers should run inside an explicit request-scoped transaction
  such as `with session_factory.begin() as session:` so reads and writes share
  one boundary.
- Worker operations should wrap each unit of work in a single explicit
  transaction so failures roll back any intermediate status updates instead of
  persisting `"processing"`.
- Admin planning and reconcile commands should also own their transaction
  boundaries. Dry-run paths should intentionally roll back queued-status changes
  while still returning their planning output.

Do not add a global session, helper-level auto-commit behavior, or a legacy
compatibility shim during the cutover.
