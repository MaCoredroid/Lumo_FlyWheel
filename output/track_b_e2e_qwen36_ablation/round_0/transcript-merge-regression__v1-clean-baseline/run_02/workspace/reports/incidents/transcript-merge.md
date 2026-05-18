# Transcript Merge Incident Note

## Root cause

`_event_key` in `replay/merge.py` keyed tool-output events by `(role, tool_name)` instead of `event_id`, causing distinct tool invocations with the same tool name to collide into a single merged block. Additionally, post-completion `debug_only` fragments were retained in the merged output, polluting rendered transcripts.

## Fix applied

- `_event_key` now returns `event_id` for tool-output events, giving each invocation stable identity.
- Post-completion `debug_only` fragments are dropped during the merge pass so they never reach rendering.
- `replay/incident_summary.py` now counts tool blocks directly from merged events (`count_source: "merged_events"`) instead of parsing rendered lines.
- `replay/render.py` was not modified; deduplication and filtering are handled by the merge layer.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
# OK
```
