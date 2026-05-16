# Transcript Merge Incident Note

## Resolution

The reducer bug has been fixed. Tool outputs now use `event_id` as stable identity,
ensuring distinct tool outputs with the same `tool_name` remain separate. Debug-only
fragments appearing after `response.completed` are now filtered out during merge.

The incident summary now counts merged tool-output blocks directly from events
rather than counting rendered lines.

## Changes

- `replay/merge.py`: Fixed `_event_key()` to use `event_id` for all events
- `replay/merge.py`: Filter `debug_only` events after completion during merge
- `replay/incident_summary.py`: Count tool blocks from merged events, not rendered lines
