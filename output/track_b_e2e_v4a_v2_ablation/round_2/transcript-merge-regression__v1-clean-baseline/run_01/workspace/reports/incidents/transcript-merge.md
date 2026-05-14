# Transcript Merge Incident Note

## Resolution

## Root cause

The reducer in `replay/merge.py` had two bugs:

1. **Unstable identity for tool outputs: the `_event_key` function grouped tool outputs by `(role, tool_name)` instead of using `event_id`, causing distinct tool outputs with the same tool name to be merged incorrectly.
2. **Post-completion debug fragments not filtered**: debug-only fragments appearing after `response.completed` were marked but not excluded from the merged result, causing them to render.

## Fix applied

- `_event_key` now uses `event_id` for all event kinds, providing stable identity.
- `merge_records` skips debug-only fragments that appear after completion.
- `render_events` filters out debug-only tool outputs as a defense-in-depth.
- `summarize_events` counts directly from merged events instead of rendered lines.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```

All tests pass.
