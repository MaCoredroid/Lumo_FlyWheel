# Transcript Merge Incident Note

## Resolution

## Fix applied

The reducer bug in `replay/merge.py` has been repaired:

- **Stable identity**: `_event_key` now uses `event_id` for tool outputs instead of `(role, tool_name)` grouping. This ensures distinct tool outputs with the same tool name remain separate.
- **Post-completion filtering**: Debug-only fragments after `response.completed` are now filtered during merge, not deferred to render.
- **Incident summary**: `replay/incident_summary.py` now counts merged events directly instead of parsing rendered lines.

## Verification

Run the test suite:

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

## Impact

- Tool outputs with identical in tool name but distinct in event_id are now correctly preserved as separate blocks.
- Debug-only fragments appearing after completion no longer leaks into rendered output.
- Incident summaries report `count_source: "merged_events"` with accurate block counts.
