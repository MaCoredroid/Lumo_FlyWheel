# Transcript Merge Incident Note

## Root cause

The reducer in `replay/merge.py` grouped `tool_output` events by
`(role, tool_name)` instead of their stable `event_id`.  Distinct tool
blocks sharing the same `tool_name` (e.g. two `fetch_weather` calls)
were collapsed into a single entry.  Debug-only fragments that arrived
after `response.completed` were merged but never filtered out, so they
leaked into rendered transcripts.

## Fix applied

- `_event_key` now uses `event_id` as the stable identity for
  `tool_output` events.
- Merged records marked `after_completion=True` (debug-only fragments
  after the completion marker) are dropped before the merge result is
  returned, so the filtering happens in the reducer, not the renderer.
- `replay/incident_summary.py` now counts tool-output blocks directly
  from merged events rather than parsing rendered lines.

## Verification

Run the test suite:

```bash
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
