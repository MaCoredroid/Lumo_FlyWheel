## Incident 2026-04-07

The prior SQLAlchemy 2 cutover was rolled back after a failed worker retry left
rows in `processing` and the admin dry-run path queued real work. The incident
was not caused by `select()` itself; it was caused by preserving legacy helper
commits across the retry and planning flows.
