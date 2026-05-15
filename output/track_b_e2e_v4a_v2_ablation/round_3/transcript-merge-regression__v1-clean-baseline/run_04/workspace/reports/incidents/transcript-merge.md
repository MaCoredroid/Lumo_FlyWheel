# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- `_event_key` now uses `event_id` for stable identity instead of `(role, tool_name)` grouping.
- Debug-only fragments after `response.completed` are filtered out during merge.
- `replay/incident_summary.py` now counts merged tool-output blocks directly.

## Verification

Run the test suite to confirm the fix:

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

## Old guidance

The previous mitigation of filtering duplicate lines at render time is no longer required.
