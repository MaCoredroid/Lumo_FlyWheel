# SQLAlchemy 2 Cutover

This cutover removes legacy `query()`-style repository access and puts
transaction ownership at the API, worker, and admin entrypoints.

## Operational changes

- Repository helpers now use SQLAlchemy 2 `select()` statements and do not
  commit on their own.
- API writes commit only at the API boundary. If an API call is already running
  inside a caller-owned transaction, the API respects that open boundary instead
  of closing it early.
- Worker settlement runs each requested unit of work inside a single
  transaction. If settlement raises after marking an entry as `processing`, the
  transaction rolls back and the entry remains at its pre-run state.
- Admin planning and reconciliation keep `dry_run=True` read-only. They still
  evaluate the rows that would be changed, but they roll back before returning.
- Admin reconciliation also rolls back the whole batch on failure instead of
  leaving the first row half-applied.

## Operator rollback guidance

- Treat `dry_run=True` output as preview-only. No ledger status change should be
  persisted after the command returns.
- If a worker batch or admin reconciliation fails, fix the upstream cause and
  retry the whole request. The failed attempt should not leave `processing` or
  `queued` residue behind.
- If post-cutover validation shows changed statuses after a dry run or failed
  batch, stop the rollout and investigate the service boundary that called the
  repository outside its owning transaction.
