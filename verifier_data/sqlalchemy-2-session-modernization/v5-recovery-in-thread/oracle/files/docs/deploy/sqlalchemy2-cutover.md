# SQLAlchemy 2 Cutover

Replace legacy `session.query(...)` access and helper-owned commits with
SQLAlchemy 2 `select(...)` calls plus caller-owned transaction boundaries.

Cutover rules:
- API write path owns its transaction and repository helpers only add/flush.
- Worker retry attempts run inside a single transaction so a failure rolls the
  entry back to `pending`.
- Admin dry-run is read-only and never queues work.

Rollback:
- redeploy the previous build
- verify no rows remain stuck in `processing` or `queued`
- rerun the worker retry and admin dry-run smoke checks before retrying

Incident follow-up:
- this cutover replaces the transaction pattern that triggered the last incident
- do not retry the rollout until the worker retry path and admin dry-run path
  both demonstrate clean rollback behavior
