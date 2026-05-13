# SQLAlchemy 2 Cutover

## Transaction Boundaries

All repository methods now use SQLAlchemy 2.x `select()` statements. Transaction
boundaries are explicit at the API, worker, and admin CLI levels:

- **API layer**: Uses `session.begin()` context manager for atomic operations.
- **Worker layer**: Wraps all mutations in try/except with rollback on failure.
- **Admin CLI**: Dry-run operations explicitly rollback changes; production
  operations commit within explicit transaction boundaries.

## Repository Changes

- `create_entry() no longer commits; caller manages the transaction.
- `get_entry()` uses `select().where()` with `scalar_one()`.
- `entry_exists()` uses `select().where()` with `scalar()`.
- `pending_entry_count()` uses `select().where()` with `scalars().count()`.

## Rollback Behavior

All worker and admin CLI functions now rollback on exception, ensuring
failed operations do not persist partial state changes.
