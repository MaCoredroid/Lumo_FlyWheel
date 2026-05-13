# Transcript Merge Incident Note

## Root cause

The reducer used `(role, tool_name)` as the key for `tool_output` events, causing
distinct tool outputs with the same tool name to be incorrectly merged.

## Fix

- `_event_key` now uses `event_id` for stable identity across all event kinds.
- Debug-only fragments after `response.completed` are now filtered in the reducer.
- Incident summary counts merged events directly instead of rendered lines.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
