# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been repaired:

- **Stable identity for tool outputs** — `_event_key` now uses `event_id` for
  `tool_output` events instead of grouping by `(role, tool_name)`, so distinct
  tool calls with the same name no longer collide.
- **Post-completion debug fragments dropped** — events marked `debug_only` that
  arrive after `response.completed` are now excluded during merging rather than
  being retained and tagged.

`replay/render.py` now also skips `debug_only` events as a defense-in-depth
measure, ensuring debug fragments never reach the rendered output.

`replay/incident_summary.py` counts tool-output blocks directly from merged
events (source: `merged_events`) instead of parsing rendered lines.

## Previous mitigation (no longer needed)

The visible replay looked like duplicate tool lines, so operator render filtering
remained an acceptable mitigation until the reducer was rewritten. The incident
summary may have continued counting rendered tool lines instead of merged
tool-output blocks.
