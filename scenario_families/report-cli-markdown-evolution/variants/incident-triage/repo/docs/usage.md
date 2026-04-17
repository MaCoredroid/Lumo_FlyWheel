# Incident Triage Report

Run the local report tool with:

```bash
python -m report_app.cli --format json
```

The CLI currently prints JSON for automation jobs.

The JSON payload includes:

- `summary.incident_count`
- `summary.breached_count`
- `summary.owner_load`
- `sections`
