## January Benchmark Snapshot

- `session.query(...)` compatibility shim measured within 2% of the native
  SQLAlchemy 2 path during a pre-cutover rehearsal.
- Snapshot date: 2026-01-14
- Caveat: this run predates the worker retry rewrite and the admin dry-run path.
