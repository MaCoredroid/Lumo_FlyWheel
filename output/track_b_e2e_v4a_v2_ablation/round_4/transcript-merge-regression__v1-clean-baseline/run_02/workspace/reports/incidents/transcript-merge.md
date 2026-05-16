# Transcript Merge Incident Note

## Root cause

The reducer in `replay/merge.py` used `(role, tool_name)` as the key for `tool_output` events, causing distinct tool outputs with the same tool name to be incorrectly merged into a single block.

## Fix applied

- `_event_key` now uses `event_id` for stable identity across all event kinds, including `tool_output`.
- Post-completion `debug_only` events are filtered out during merge, not after rendering.
- `incident_summary.py` now counts from merged events directly instead of rendered lines.

## Verification

Run: `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`
