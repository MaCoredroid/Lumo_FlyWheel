# Transcript Merge Incident Note

## Resolution

The reducer bug in `replay/merge.py` has been fixed:

- `_event_key` now uses `event_id` for stable identity instead of role/name grouping
- Post-completion debug-only fragments are filtered during merge, not after rendering
- `replay/render.py` filters `debug_only` and `after_completion` events
- `replay/incident_summary.py` counts merged events directly

## Old guidance

Until the reducer is rewritten, the incident summary may continue counting
rendered tool lines instead of merged tool-output blocks.
