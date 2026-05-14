# SQLAlchemy 2 Cutover

Repository access now uses SQLAlchemy 2.x `select()` syntax instead of legacy
`.query()` style. The API, worker, and admin CLI each have explicit transaction
boundaries with correct rollback behavior:

- **API**: Operations run within caller-provided transaction boundaries; the
  repository no longer commits independently.
- **Worker**: `settle_entry` and `settle_batch` commit only on success; any
  exception triggers rollback, leaving entries in their prior state.
- **Admin CLI**: `plan_batch` and `reconcile_batch` with `dry_run=True` perform `rollback()`
  when `dry_run=True`, ensuring no status changes persist.

Legacy compatibility shims have been removed. All transaction boundaries are
now explicit and follow SQLAlchemy 2.x semantics.
