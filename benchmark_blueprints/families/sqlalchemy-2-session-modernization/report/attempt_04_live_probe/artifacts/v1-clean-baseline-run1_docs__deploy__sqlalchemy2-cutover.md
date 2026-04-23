# SQLAlchemy 2 Cutover

Move repository access to SQLAlchemy 2-style `select(...)` statements and keep
repository helpers side-effect free with respect to transactions. `create_*`
helpers may `flush()` to surface constraint errors, but they must not commit.

Own transaction boundaries at the service entrypoints:

- API writes should open a transaction around the request-level mutation. If the
  caller already started a transaction, use a nested boundary instead of
  committing the outer unit of work.
- Worker settlement flows should wrap the full job or batch in one transaction
  so any exception rolls back intermediate `"processing"` updates.
- Admin CLI commands should explicitly commit only non-dry-run changes and
  explicitly roll back dry runs or failed batches.

This preserves SQLAlchemy 2.x semantics without a legacy compatibility shim, a
global session, or helper-level commits that can leak partial state.
