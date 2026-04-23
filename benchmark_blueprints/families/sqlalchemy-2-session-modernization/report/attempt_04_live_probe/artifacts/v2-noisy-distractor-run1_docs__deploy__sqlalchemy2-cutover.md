# SQLAlchemy 2 Cutover

Repository access now uses SQLAlchemy 2-style `select(...)` statements, and
repository helpers no longer call `commit()` on behalf of callers.

Transaction ownership is explicit at each boundary:

- API requests supply a session/transaction from the caller and can compose
  multiple repository calls in one unit of work.
- Worker settlement runs each entry or batch inside a single transaction and
  rolls back the full operation if an exception is raised before commit.
- Admin planning/reconciliation uses one transaction per command; dry runs roll
  back status changes instead of persisting preview state.
