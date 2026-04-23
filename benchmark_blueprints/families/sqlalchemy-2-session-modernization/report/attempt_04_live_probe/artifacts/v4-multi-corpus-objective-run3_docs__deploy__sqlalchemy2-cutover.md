# SQLAlchemy 2 Cutover

This cutover moves `ledger-sync` to SQLAlchemy 2-style repository access and
service-owned transaction boundaries. Repository helpers no longer commit on
their own; the API, worker, and admin CLI each define the transaction scope for
their own unit of work so rollback behavior is predictable during failures.

## Operational Expectations

- API writes should run inside an explicit request transaction. Reads may use a
  short-lived session, but repository helpers must stay side-effect free.
- Worker settlement uses one transaction per settle operation or batch. If a
  retry path raises after marking an entry in memory, the whole unit of work is
  rolled back and the row remains at its pre-run state.
- Admin planning keeps `dry_run=True` read-only by rolling back after staging
  the candidate status changes. A non-dry-run batch commits only once after the
  full batch succeeds.

## Rollback Playbook

1. Stop issuing new admin reconcile or worker settle commands.
2. Re-run the failed worker or CLI command only after confirming the prior run
   exited before commit; SQLAlchemy 2 transaction scoping now guarantees that a
   mid-run exception leaves the database unchanged for that unit of work.
3. Use the admin CLI in `dry_run=True` mode first when validating a production
   batch, because the dry run exercises the same selection/update path while
   rolling back before commit.
4. If operators need to validate state after an interrupted release, inspect
   the affected entries directly; do not add a compatibility shim or ad hoc
   repository-level commit to force legacy behavior.
