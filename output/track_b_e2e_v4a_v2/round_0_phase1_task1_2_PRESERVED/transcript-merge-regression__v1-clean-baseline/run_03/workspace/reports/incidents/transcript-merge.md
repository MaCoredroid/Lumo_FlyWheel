# Transcript Merge Incident Note

## Root cause

The reducer in `replay/merge.py` incorrectly grouped tool outputs by `(role, tool_name)` instead of using `event_id` for stable identity. This caused distinct tool outputs with the same tool name to be merged incorrectly.

Additionally, debug-only fragments after `response.completed` were not being dropped, causing them to appear in rendered output.

## Resolution

- `_event_key` now uses `event_id` for all event types, ensuring stable identity independent of fixture order.
- Post-completion debug-only fragments are now dropped during merge, not filtered at render time.
- `incident_summary.py` now counts merged events directly instead of parsing rendered lines.

## Verification

Run: `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`
