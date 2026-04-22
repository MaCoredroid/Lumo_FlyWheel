# Nightly Regression Watch Runbook

## Regenerate the digest

Run:

```bash
python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --out ops_digest/fixtures/inbox/generated_digest.md
```

## Escalation contract

- Use the literal `Action required` label only for blocking nights: final verdict failed or a required milestone is missing.
- Select the latest completed run for each `report_date`; replace stale same-day reports instead of appending duplicates.
- Repair the existing `nightly_regression_watch.toml` automation in place.
