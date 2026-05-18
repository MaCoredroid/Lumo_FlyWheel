# Transcript Merge Incident Note

## Resolution

The reducer in `replay/merge.py` has been repaired:

- `_event_key` now derives stable identity from `event_id` for all event kinds
  (including `tool_output`), so same-name tool outputs no longer collide.
- Post-completion `debug_only` fragments are dropped during merge rather than
  leaking through to render.

`replay/incident_summary.py` now counts tool-output blocks directly from the
merged event list (`count_source: merged_events`) instead of parsing rendered
lines.

## Old guidance (superseded)

The visible replay looked like duplicate tool lines, so operator render
filtering was an acceptable mitigation. The incident summary counted rendered
tool lines instead of merged tool-output blocks. Both workarounds are no longer
needed.
