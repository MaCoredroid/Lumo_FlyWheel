# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- `_event_key` now uses `event_id` for stable identity on all event kinds, including `tool_output`. Same-named tools no longer collide.
- Debug-only fragments that arrive after a `response.completed` event are dropped during merge, so they no longer leak into rendered output.

`replay/incident_summary.py` now counts tool-output and assistant blocks directly from merged events (`count_source: "merged_events"`) instead of parsing rendered lines.

## Previous mitigation (no longer needed)

Operator render filtering was used to hide duplicate tool lines. This is no longer required since the reducer correctly deduplicates by `event_id`.
