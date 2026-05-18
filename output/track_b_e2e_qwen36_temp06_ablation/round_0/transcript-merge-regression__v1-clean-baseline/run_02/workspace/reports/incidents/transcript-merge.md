# Transcript Merge Incident Note

## Root cause

`replay/merge.py` had two bugs in the reducer:

1. `_event_key` grouped `tool_output` events by `(role, tool_name)` instead of `event_id`, collapsing distinct tools with the same name into a single merged record.
2. Debug-only fragments appearing after `response.completed` were kept in the merged output rather than discarded, causing them to leak into rendered transcripts.

## Fix applied

- `_event_key` now returns `event_id` for `tool_output` events, giving each tool call stable identity.
- Merged records marked `after_completion` (debug-only fragments post-completion) are filtered out before the reducer returns.
- `replay/incident_summary.py` now counts tool/assistant blocks directly from merged events (`count_source: "merged_events"`) instead of parsing rendered lines.

## Verification

All tests pass:

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
