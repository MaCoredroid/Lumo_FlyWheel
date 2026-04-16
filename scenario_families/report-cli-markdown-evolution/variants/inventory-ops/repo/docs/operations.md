# Operations Notes

- Nightly automation still ingests JSON from `python -m report_app.cli`.
- Human handoff screenshots come from the same CLI, so the Markdown output
  must stay stable between runs.
- Owner totals are sorted by queue count descending and owner name
  ascending to keep screenshots easy to compare across shifts.
