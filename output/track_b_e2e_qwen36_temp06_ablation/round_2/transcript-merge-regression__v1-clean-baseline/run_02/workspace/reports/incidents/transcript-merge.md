# Transcript Merge Incident Note

## Root cause

The reducer in `replay/merge.py` had two bugs:

1. **Unstable event identity for `tool_output` events — the `_event_key` function
grouped by `(role, tool_name)` instead of `event_id`, causing distinct tool
outputs with the same tool name to collapse into a single merged block.

2. **Post-completion debug fragments not dropped** — debug-only fragments
arriving after `response.completed` were tagged but not discarded, so they
survived into the rendered output.

## Fixes applied

- `_event_key` now uses `event_id` for stable identity across all event kinds.
- `merge_records` skips debug-only events that arrive after completion.
- `reports/incident_summary.py` now counts directly from merged events
  (`count_source: "merged_events"`) instead of parsing rendered lines.

## Validation

All tests pass:

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
