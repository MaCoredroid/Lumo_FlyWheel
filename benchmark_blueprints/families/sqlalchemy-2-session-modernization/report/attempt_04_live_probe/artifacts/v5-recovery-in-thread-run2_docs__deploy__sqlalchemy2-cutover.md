# SQLAlchemy 2 Cutover

This cutover removes the legacy helper-owned commit pattern that caused the
2026-04-07 rollback. SQLAlchemy 2-style repository access now stays
transaction-neutral, while the API, worker, and admin CLI each own their
operation boundary explicitly.

## Behavioral changes

- Repository reads use SQLAlchemy 2 `select()` execution paths.
- Repository writes no longer call `commit()`; callers decide when work is
  committed or rolled back.
- Worker settle flows run in a single transaction per operation or batch, so a
  failure after marking `processing` rolls the row back to its pre-retry state.
- Admin dry-run paths never commit queue-state changes. Dry-run planning can
  inspect and stage work inside the session, but it must roll back before
  returning.

## Release gate checks

- Force a worker failure after the retry mark and verify affected rows remain
  `pending`.
- Run admin dry-run planning and verify it does not leave rows in `queued`.
- Verify batch failures roll back the entire unit of work instead of preserving
  partial `processing` or `queued` state.

## Rollback and recovery

If the cutover must be rolled back, stop new worker/admin runs first, then
confirm no rows are stranded in `processing` or `queued` from an in-flight
attempt. The expected recovery state after a failed retry or dry-run is the
pre-operation state, typically `pending`.

Before retrying the release:

1. Re-run the worker forced-failure check and confirm rollback leaves rows at
   `pending`.
2. Re-run the admin dry-run check and confirm it stays read-only.
3. Only resume traffic once both rollback proofs pass and on-call has the exact
   commands needed to repeat them.
