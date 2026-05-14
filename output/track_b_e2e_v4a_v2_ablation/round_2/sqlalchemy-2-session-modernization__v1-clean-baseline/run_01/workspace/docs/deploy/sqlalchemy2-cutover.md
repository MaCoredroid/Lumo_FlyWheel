# SQLAlchemy 2 Cutover

## Changes Made

- Repository layer now uses SQLAlchemy 2.x `select()` syntax instead of legacy `.query()`
- All transaction boundaries are explicit using `session_factory.begin()` context manager
- Repository helpers no longer commit on their own - transactions are managed at the API/worker/CLI level
- Rollback behavior is correct: exceptions within a `begin()` block automatically rollback

## Migration Notes

- API layer: Uses `session_factory.begin()` for explicit transaction boundaries
- Worker layer: Each entry settlement uses its own transaction for proper isolation
- Admin CLI: Dry runs do not persist changes (transaction rolls back), real runs commit
