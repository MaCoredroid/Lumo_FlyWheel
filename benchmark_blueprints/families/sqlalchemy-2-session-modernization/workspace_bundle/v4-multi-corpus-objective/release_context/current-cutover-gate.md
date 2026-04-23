## Release Gate

Blocking check for this cutover:
- Admin dry-run must stay read-only on production-like data.
- Batch settlement retries must roll back cleanly if the first row fails.
- Operators need an explicit rollback playbook, not a compatibility shim note.
