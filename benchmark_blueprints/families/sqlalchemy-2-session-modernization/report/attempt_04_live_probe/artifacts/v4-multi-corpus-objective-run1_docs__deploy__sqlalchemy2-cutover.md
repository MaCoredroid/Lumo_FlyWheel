# SQLAlchemy 2 Cutover

This cutover removes helper-owned commits and moves transaction ownership to the
API, worker, and admin command surfaces. Repository access now uses
SQLAlchemy 2-style `select()` execution so callers can control one transaction
per unit of work and get predictable rollback behavior.

## Operator Expectations

- API writes should run inside an explicit request-scoped transaction. Reads may
  use a plain session, but they must not rely on repository helpers to commit or
  close work implicitly.
- Worker settlement retries are now atomic per call. If settlement fails after a
  row is marked `processing`, the worker transaction rolls back and the entry
  returns to its pre-run state.
- Admin dry runs are read-only by design. Planning or reconciliation commands
  may stage status changes in memory, but they must always roll back before the
  command returns when `dry_run=True`.

## Rollback Playbook

1. Stop new worker or admin invocations if operators observe unexpected status
   transitions during cutover.
2. Re-run the failed worker or admin command only after confirming the previous
   attempt rolled back and entries remain in their pre-command state.
3. If a non-dry-run batch fails, expect the whole transaction to roll back; do
   not manually repair partially queued or partially settled rows because the
   cutover no longer commits mid-batch.

Do not restore a legacy compatibility shim as a mitigation. The release gate for
this cutover depends on dry-run safety and failure-time rollback semantics.
