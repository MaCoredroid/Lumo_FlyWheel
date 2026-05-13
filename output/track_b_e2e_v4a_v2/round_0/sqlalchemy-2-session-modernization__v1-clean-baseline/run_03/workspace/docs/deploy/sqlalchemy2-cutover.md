# SQLAlchemy 2 Cutover

## Summary

The ledger-sync service has been modernized to SQLAlchemy 2.x semantics with proper
transaction boundaries.

## Changes

- **Repository layer**: Migrated from `.query()` to `select()` syntax. The `create_entry`
  function no longer commits; callers manage their own transactions.

- **Worker**: `settle_entry` and `settle_batch` now commit only after all work completes.
  Exceptions trigger automatic rollback, so partial state changes are not persisted.

- **Admin CLI**: `plan_batch` with `dry_run=True` now rolls back instead of committing.
  `reconcile_batch` commits only when `dry_run=False` and no exception occurred.

## Transaction Behavior

- API operations use explicit `session.begin()` context for transactional boundaries.
- Worker and admin CLI functions manage transactions internally with proper rollback.
- No legacy compatibility shim is in place; all code uses SQLAlchemy 2.x patterns.
