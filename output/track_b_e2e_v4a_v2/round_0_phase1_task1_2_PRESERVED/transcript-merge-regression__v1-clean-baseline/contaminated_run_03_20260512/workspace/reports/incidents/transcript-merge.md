# Transcript Merge Incident Note

## Resolution

The reducer bug has been fixed. Tool outputs now use `event_id` for stable identity
instead of `(role, tool_name)` grouping. Debug-only fragments after completion are
properly filtered during rendering. The incident summary now counts directly from
merged events.

## Changes

- `replay/merge.py`: Fixed `_event_key` to use `event_id` for all event types
- `replay/render.py`: Added filter for `debug_only` + `after_completion` events
- `replay/incident_summary.py`: Now counts from merged events, not rendered lines
