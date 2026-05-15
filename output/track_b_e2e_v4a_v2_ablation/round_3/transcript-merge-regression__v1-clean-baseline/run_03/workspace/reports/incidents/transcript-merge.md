# Transcript Merge Incident Note

## Root Cause

The reducer in `replay/merge.py` used `(role, tool_name)` as the key for tool_output
events, causing distinct tool outputs with the same tool name to be incorrectly
merged into a single block. Additionally, debug-only fragments appearing after
`response.completed` were not filtered during the merge phase, allowing them to
survive into the rendered output.

## Fix

- `_event_key` now uses `event_id` for stable identity of all events, including
  tool outputs, rather than grouping by role/name.
- `merge_records` now skips `debug_only` events that appear after a completion
  marker, ensuring they are filtered at the merge phase rather than relying on render-time filtering.
- `incident_summary.py` now counts directly from merged events instead of
  parsing rendered output, with `count_source` set to `"merged_events"`.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
