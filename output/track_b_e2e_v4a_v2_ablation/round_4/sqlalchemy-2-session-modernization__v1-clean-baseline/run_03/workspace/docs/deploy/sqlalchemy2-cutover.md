# SQLAlchemy 2 Cutover

## Changes

- Repository access now uses SQLAlchemy 2.x `select()` syntax instead of legacy `query()`.
- API, worker, and admin CLI have explicit transaction boundaries with correct rollback behavior.
- Repository helpers no longer commit on their own; callers manage transactions.

## Migration Notes

- All session usage should use `with session_factory() as session:` context manager.
- Explicit `session.commit()` is required after writes.
- On failure, the context manager automatically rolls back uncommitted changes.
- Dry-run operations explicitly restore original values and call `session.rollback()`.
