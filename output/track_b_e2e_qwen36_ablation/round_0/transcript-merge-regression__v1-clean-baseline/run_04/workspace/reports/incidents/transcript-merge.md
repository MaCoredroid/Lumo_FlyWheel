# Transcript Merge Incident Note

## Fix applied

The reducer (`replay/merge.py`) now uses `event_id` as the stable key for
`tool_output` events instead of `(role, tool_name)`, so tools with the same
name remain distinct.

Post-completion debug-only fragments are filtered at the merge level
(`after_completion=True`) and excluded by the renderer — no post-render
deduplication is needed.

The incident summary (`replay/incident_summary.py`) counts directly from
merged events (`count_source: "merged_events"`) instead of parsing rendered
lines.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
