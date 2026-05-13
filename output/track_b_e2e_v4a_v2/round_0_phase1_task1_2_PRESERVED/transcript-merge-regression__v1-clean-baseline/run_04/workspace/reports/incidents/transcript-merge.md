# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- **Stable identity**: Tool outputs now use `event_id` for stable identity instead of `(role, tool_name)` grouping.
- **Post-completion filtering**: Debug-only fragments after `response.completed` are now properly dropped during merge.
- **Incident summary**: Now counts merged tool-output blocks directly from `merged_events` instead of rendered lines.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

## Old guidance (deprecated)

Previously, render filtering was used as a mitigation for duplicate tool lines. This is no longer needed as the reducer now handles deduplication correctly.
