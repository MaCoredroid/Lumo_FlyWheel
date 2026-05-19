# Transcript Merge Incident Note

## Root Cause

`_event_key` in `replay/merge.py` grouped `tool_output` events by
`(role, tool_name)` instead of their `event_id`, causing distinct tool
outputs with the same name to collapse into a single merged record.

Additionally, debug-only fragments that arrived after `response.completed`
were marked but never dropped, so they leaked into the rendered transcript.

## Fix

- `_event_key` now returns `event_id` (with a fallback on `tool_name` +
  `sequence`) for `tool_output` events, giving each tool output stable
  identity.
- `merge_records` skips debug-only events that arrive after completion at the
  reducer level so they never reach render.
- `incident_summary.py` now counts merged tool-output blocks directly
  instead of parsing rendered lines.

## Files changed

- `replay/merge.py` – fixed `_event_key` and post-completion filtering.
- `replay/incident_summary.py` – count merged events, not rendered lines.
