
# Escalation Summary

Support note from 2026-04-10:

- `owner_source` in the exported project-board payload looks like it comes from storage because the record already has `owner` when it comes back.
- `routing_key` is probably computed before the CLI applies `--owner`.
- Please confirm whether the fix belongs in the storage layer or only in docs.
