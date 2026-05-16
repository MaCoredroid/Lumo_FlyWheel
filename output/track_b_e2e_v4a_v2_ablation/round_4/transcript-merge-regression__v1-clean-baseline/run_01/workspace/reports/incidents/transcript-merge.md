# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- **Stable event identity**: Tool outputs now use `event_id` for stable identity instead of `(role, tool_name)` grouping. Distinct tool events with the same `tool_name` remain separate.
- **Post-completion filtering**: Debug-only fragments after `response.completed` are now filtered during merge, not at render time.
- **Incident summary**: `replay/incident_summary.py` now counts directly from merged events (`count_source: "merged_events"`), not rendered lines.

## Verification

Run the test suite to confirm the fix:

```bash
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

## Old guidance

The previous mitigation of filtering duplicate tool lines at render time is no longer needed. The reducer now correctly handles event identity and post-completion fragments.
