# Transcript Merge Incident Note

## Resolution

## Root cause

The reducer in `replay/merge.py` used `(role, tool_name)` as the stable identity for tool_output events, causing distinct tool outputs with the same tool name to be incorrectly merged into a single block.

## Fix applied

- `_event_key()` now uses `event_id` as the stable identity for all events, including tool_output events.
- Debug-only fragments appearing after `response.completed` are now filtered out during merge, not just marked.
- `incident_summary.py` now counts merged events directly instead of counting rendered lines.

## Verification

Run the following to confirm the fix:

```bash
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

All three tests should pass.
