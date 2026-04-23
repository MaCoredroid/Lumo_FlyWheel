# SQLAlchemy 2 Cutover

This cutover removes the legacy helper-owned commits that caused the
2026-04-07 rollback. The SQLAlchemy 2 migration is only considered ready when
repository reads use `select(...)` and each caller owns a single explicit
transaction boundary for its full unit of work.

## Behavioral changes

- Repository helpers no longer call `commit()`. They add, flush, and read
  rows inside the caller's transaction.
- Worker settlement runs in one explicit transaction per call. A forced failure
  after marking `processing` must roll back to `pending` so retries start clean.
- Admin planning and reconciliation own their transaction boundaries as well.
  Dry-run paths must roll back instead of queueing real work.

## Recovery checks before release

- Force `settle_entry(..., fail_after_mark=True)` and verify the row is still
  `pending`.
- Force admin dry-run planning and verify no row remains `queued`.
- Force a batch failure and verify the entire batch rolls back instead of
  persisting partial retry state.

## Rollback steps

1. Stop new worker and admin runs.
2. Revert the application deploy to the previous release.
3. Inspect `ledger_entries` for rows left in `processing` or `queued` from the
   failed attempt.
4. Reset those rows to `pending` before re-enabling retries.
5. Re-run the recovery checks above before attempting the cutover again.
