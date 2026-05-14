# Transcript Merge Incident Note

## Resolution

The reducer bug has been fixed. Tool outputs now use `event_id` for stable identity,
ensuring distinct tool outputs remain separate even when they share the same tool name.
Post-completion debug-only fragments are now filtered out during merge, not at render time.

## Changes

- `replay/merge.py`: `_event_key` now uses `event_id` for all event kinds, including `tool_output`.
- `replay/merge.py`: Post-completion `debug_only` fragments are filtered during merge.
- `replay/incident_summary.py`: Now counts merged events directly instead of rendered lines.

## Verification

Run the following to confirm the fix:

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
