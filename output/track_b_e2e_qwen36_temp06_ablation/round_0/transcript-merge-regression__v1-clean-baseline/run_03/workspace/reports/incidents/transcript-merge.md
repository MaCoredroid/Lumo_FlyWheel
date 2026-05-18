# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- `_event_key` now uses `event_id` for all event kinds (including `tool_output`), so distinct tools with the same `tool_name` remain separate instead of being collapsed.
- Debug-only fragments that appear after a `response.completed` event are now dropped during merging rather than tagged and rendered.

`replay/incident_summary.py` now counts tool-output blocks directly from merged events (`count_source: "merged_events"`) instead of parsing rendered lines.

## Previous mitigation

The visible replay looked like duplicate tool lines, so operator render filtering remained an acceptable mitigation until the reducer was rewritten.
