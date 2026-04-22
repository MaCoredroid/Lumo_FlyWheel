# Verification

- `PYTHONPATH=src python -m unittest tests.test_cli tests.test_service tests.test_docs`
- `PYTHONPATH=src python -m watchlist_report.cli --format json`
- `PYTHONPATH=src python -m watchlist_report.cli --format markdown --include-watchlist`

Observed:

- markdown output now includes `## Watchlist Follow-Up`
- baseline JSON output still matches `tests/fixtures/json/baseline_report.json`
- `tests/fixtures/legacy_snapshot.md` remained untouched
