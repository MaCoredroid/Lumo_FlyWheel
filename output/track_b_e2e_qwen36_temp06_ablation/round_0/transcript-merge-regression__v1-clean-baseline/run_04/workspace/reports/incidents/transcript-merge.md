# Transcript Merge Incident Note

## Resolution

The reducer in `replay/merge.py` has been repaired:

- **Tool-output identity**: `_event_key` now uses `event_id` for `tool_output` events instead of grouping by `(role, tool_name)`, so distinct tool calls with the same tool name remain separate blocks.
- **Post-completion debug fragments**: Events marked `debug_only` that arrive after `response.completed` are now dropped during merge rather than being carried through to render.

`replay/incident_summary.py` now counts directly from merged events (`count_source: "merged_events"`) instead of parsing rendered lines.

## Previous mitigation

The visible replay previously showed duplicate tool lines, so operator render filtering was used as a temporary workaround. This is no longer needed.
