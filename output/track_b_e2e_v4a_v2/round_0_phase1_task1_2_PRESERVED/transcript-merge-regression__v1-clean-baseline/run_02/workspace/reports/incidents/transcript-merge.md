# Transcript Merge Incident Note

## Resolution

The reducer bug has been fixed in `replay/merge.py`:

- `_event_key` now uses `event_id` for stable identity instead of `(role, tool_name)` grouping.
- Post-completion `debug_only` fragments are dropped during merge, not just marked.

The `replay/render.py` module remains unchanged - it stays honest by not filtering duplicates.

The `replay/incident_summary.py` now counts from merged events directly, reporting `count_source: "merged_events"`.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

