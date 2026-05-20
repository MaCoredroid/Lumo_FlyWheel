# Transcript Merge Incident Note

## Resolution

The reducer in `replay/merge.py` has been fixed:

- **Stable event identity**: `_event_key` now uses `event_id` for all event types
  (including `tool_output`), so distinct tool outputs with the same `tool_name`
  no longer collapse into a single merged block.
- **Post-completion debug fragments**: Events marked `debug_only` that arrive after
  a `response.completed` sentinel are now dropped at merge time instead of being carried
  through to render.

`replay/incident_summary.py` now counts blocks directly from merged events
(`count_source: "merged_events"`) rather than from rendered lines.

## Previous mitigation (no longer needed)

The visible replay looked like duplicate tool lines, so operator render filtering was
used as a stopgap. The reducer fix makes this unnecessary.
