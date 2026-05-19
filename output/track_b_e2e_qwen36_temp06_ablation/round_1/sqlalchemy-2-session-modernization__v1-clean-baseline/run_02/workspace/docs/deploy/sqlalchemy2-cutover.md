# SQLAlchemy 2 Cutover

Repository queries have been migrated to SQLAlchemy 2.x `select()` semantics.
The legacy `session.query()` pattern is no longer used.

Transaction boundaries are now explicit at the call site (API, worker, and
admin CLI). Repository helpers no longer commit internally — the caller owns
the transaction lifecycle.

Key changes:

- **Repository** — uses `session.execute(select(...))` instead of
  `session.query()`. `create_entry` no longer calls `session.commit()`.
- **API** — callers pass a session managed by `session_factory.begin()`;
  the transaction commits or rolls back at the context-manager boundary.
- **Worker** — `settle_entry` wraps the full mark-and-settle flow in a single
  `session_factory.begin()` block so failures roll back the intermediate
  "processing" status. `settle_batch` processes each entry in its own
  transaction so one failure doesn't leave others half-settled.
- **Admin CLI** — `plan_batch` rolls back when `dry_run=True` so no status
  change persists. `reconcile_batch` uses per-entry transactions with correct
  rollback on dry runs.
