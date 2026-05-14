# SQLAlchemy 2 Cutover

The ledger-sync service has been modernized to use SQLAlchemy 2.x semantics.

## Key Changes

### Repository Layer
- Migrated from legacy `.query()` API to SQLAlchemy 2.x `select()` statements
- Removed blanket `session.commit()` calls from repository helpers
- Repository functions now operate within caller-managed transactions

### Transaction Boundaries
- **API layer**: Uses `session_factory.begin()` context manager for explicit transaction boundaries
- **Worker layer**: Each settlement operation uses a single transaction with `session.commit()` at the end; exceptions trigger automatic rollback
- **Admin CLI**: Dry-run operations explicitly rollback; production operations commit at the end

### Rollback Behavior
- All operations now use proper try/commit patterns or single-transaction semantics
- Exceptions during processing trigger automatic rollback via session context manager
- No partial commits persist on failure
