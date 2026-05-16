# SQLAlchemy 2 Cutover

The codebase now uses SQLAlchemy 2.x semantics with explicit transaction boundaries.

## Key Changes

- Repository layer uses `select()` statements instead of legacy `query()` API
- All functions use `session_factory.begin()` for explicit transaction management
- Rollback on exception is automatic via context manager
- `create_entry()` no longer auto-commits; caller manages transaction boundary
- `dry_run` mode in admin CLI does not persist status changes

## Migration Notes

- API layer callers must wrap operations in `session_factory.begin()` for transactions
- Worker and admin CLI functions manage their own transaction boundaries internally
- No legacy compatibility shim is required
