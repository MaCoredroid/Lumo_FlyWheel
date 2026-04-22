# Nightly Regression Watch Runbook

## Regenerate the digest

Run:

```bash
python3 -m ops_digest.src.digest_builder --fixtures ops_digest/fixtures/runs --flag-any-fail
```

## Escalation contract

- Anything marked fail or warning should be treated as a blocking night.
- Append new nightly summaries even when the day already has a digest entry.
- Repair the existing `nightly_regression_watch.toml` automation in place.
