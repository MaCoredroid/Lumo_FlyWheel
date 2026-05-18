# Transcript Merge Incident Note

## Resolution

The reducer in `replay/merge.py` has been rewritten:

- `_event_key` now derives stable identity from `event_id` for all event kinds, including `tool_output`. Previously it grouped tool outputs by `(role, tool_name)`, which collapsed distinct tool calls sharing the same tool name into a single block.
- `merge_records` now drops `debug_only` fragments that appear after `response.completed`, preventing them from leaking into rendered output.

`replay/incident_summary.py` now counts tool-output blocks directly from merged events (`count_source: "merged_events"`) instead of scanning rendered lines.

Operator render filtering is no longer required.

## Old guidance (superseded)

Until the reducer is rewritten, the incident summary may continue counting
rendered tool lines instead of merged tool-output blocks.
