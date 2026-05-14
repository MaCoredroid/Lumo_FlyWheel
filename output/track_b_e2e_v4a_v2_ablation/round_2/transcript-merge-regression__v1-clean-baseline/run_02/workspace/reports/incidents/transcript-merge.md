# Transcript Merge Incident Note

## Root cause

The reducer in `replay/merge.py` used `(role, tool_name)` as the event key for `tool_output` events, causing distinct tool outputs with the same tool name to be incorrectly merged into a single block.

## Fix applied

- `_event_key` now uses `event_id` for all event kinds, ensuring stable identity based on the event's unique identifier rather than grouping by role/name.
- `debug_only` events are now filtered during merge, preventing debug fragments from appearing in rendered output.
- `incident_summary.py` now counts directly from merged events instead of rendered lines.

## Verification

Run: `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`
