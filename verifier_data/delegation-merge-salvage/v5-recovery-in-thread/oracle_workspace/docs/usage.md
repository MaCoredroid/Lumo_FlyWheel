# Usage

Generate the report in JSON:

```bash
PYTHONPATH=src python -m watchlist_report.cli --format json
```

Generate a readable markdown report:

```bash
PYTHONPATH=src python -m watchlist_report.cli --format markdown --include-watchlist
```
