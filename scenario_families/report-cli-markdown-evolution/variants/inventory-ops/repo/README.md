# Inventory Ops Report

`inventory-ops-report` is a small shift-handoff tool used by warehouse
operations. The CLI collects the active queue records, builds an owner
summary, and renders a report for either automation or humans reviewing
the next shift.

The repo intentionally keeps the core queue math separate from rendering:

- `report_app.service` owns the runtime entrypoints used by the CLI.
- `report_app.summaries` turns raw queue records into owner totals.
- `report_app.markdown` contains the human handoff layout.
- `report_app.rendering` contains the stable JSON output used by the
  overnight automation jobs.

The current production job still expects JSON, so any Markdown work must
leave that path unchanged.
