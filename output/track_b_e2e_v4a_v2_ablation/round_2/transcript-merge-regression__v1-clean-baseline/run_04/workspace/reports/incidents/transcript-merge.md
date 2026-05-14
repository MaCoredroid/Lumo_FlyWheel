# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- **Stable identity**: `_event_key` now uses `event_id` for `tool_output` events instead of `(role, tool_name)` grouping, ensuring distinct tool outputs with the same tool name remain separate.
- **Post-completion filtering**: `debug_only` fragments after `response.completed` are now filtered out during merge, not just at render time.
- **Incident summary**: `summarize_events` now counts from merged events directly (`count_source: "merged_events"`) instead of rendered lines.

## Verification

Run the test suite to confirm the fix:

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
