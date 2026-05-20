# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- **Stable event identity** — `_event_key` now uses `event_id` for all event kinds (including `tool_output`), so distinct tool calls with the same `tool_name` are no longer collapsed into one.
- **Post-completion debug fragments** — events marked `debug_only` that arrive after `response.completed` are now removed from the merged list instead of being retained with a flag.
- **Incident summary** — `replay/incident_summary.py` now reports `count_source` as `"merged_events"` and counts tool-output blocks directly from the merged event list.

## Previous mitigation

The visible replay showed duplicate tool lines, so operator render filtering was used as a temporary mitigation. This is no longer needed.
