# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- Tool outputs now derive stable identity from `event_id` instead of `(role, tool_name)` grouping.
- Post-completion `debug_only` fragments are filtered out during merge, not after rendering.
- Incident summary now counts merged tool-output blocks directly from `merged_events`.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

## Old guidance

Until the reducer is rewritten, the incident summary may continue counting
rendered tool lines instead of merged tool-output blocks.
