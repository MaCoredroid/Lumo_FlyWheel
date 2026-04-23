# SQLAlchemy 2 Cutover

This cutover removes legacy ORM query patterns and pushes transaction control
to the service boundary.

- Repository helpers use SQLAlchemy 2-style `select(...)` statements and never
  call `commit()` on behalf of their callers.
- API request handlers should run inside an explicit caller-managed transaction
  so repository writes and follow-up reads stay in the same unit of work.
- Worker settlement and admin reconciliation flows open one transaction per
  operation with `session_factory.begin()`. Any exception rolls back the full
  status transition instead of leaving partially committed `processing` or
  `queued` rows behind.
- Dry-run admin commands must avoid persisting status changes. They should read
  the target rows and return the planned work without mutating committed state.
