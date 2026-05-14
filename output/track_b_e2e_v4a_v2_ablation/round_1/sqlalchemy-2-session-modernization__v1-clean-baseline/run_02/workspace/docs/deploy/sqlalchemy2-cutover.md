# SQLAlchemy 2 Cutover

## Transaction Boundaries

All repository operations now use SQLAlchemy 2.x `select()` statements instead of
legacy `query()` API. Transaction boundaries are explicit in each layer:

- **API layer**: Operations**: The API layer delegates to repository functions without
  committing. Callers must manage their own transaction boundaries using
  `session_factory.begin()` context manager.

- **Worker Operations**: Each worker function manages its own transaction with
  proper rollback on failure. If an exception occurs, changes are rolled back.

- **Admin CLI**: Dry-run operations explicitly rollback changes. Production
  operations commit only on success.

## Migration Notes

- Repository helpers no longer commit on their own
- All queries use `select()` with explicit `where()` clauses
- Session context managers ensure proper cleanup on exceptions
- Dry-run operations use `rollback()` to discard changes
