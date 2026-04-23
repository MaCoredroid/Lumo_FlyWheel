# SQLAlchemy 2 Cutover

This cutover removes legacy helper-level commits and moves transaction control to
the API, worker, and admin command boundaries. Repository reads use SQLAlchemy
2-style `select(...)` statements so helper access stays side-effect free while
the call sites decide when to commit or rollback.

## Recovery Expectations

- A failed worker retry must rollback the transient `processing` mark and leave
  the row at `pending`.
- Admin dry-run flows must stay read-only and must not queue rows.
- Batch failures must rollback the whole in-flight command so retry state does
  not leak across entries.

## Rollout Checks

Before release, run `pytest -q tests/test_api.py tests/test_worker.py
tests/test_admin_cli.py` and confirm:

- API writes can participate in an outer transaction without closing it early.
- Forced worker failures recover to `pending`.
- Admin dry-runs leave existing rows unchanged.

## Rollback Notes

If rollback is required during deployment, stop new worker/admin traffic first,
redeploy the prior build, and verify impacted rows are back to `pending` before
re-enabling retries. The prior incident was caused by retry-state leakage from
helper-owned commits, so operator verification should focus on clearing any
stranded `processing` or accidentally `queued` rows before resuming traffic.
