# Transcript Merge Incident Note

## Root Cause

## Summary

The reducer incorrectly grouped tool outputs by `(role, tool_name)` instead of
using `event_id` for stable identity. This caused distinct tool outputs with
the same tool name to be merged into a single block.

Additionally, debug-only fragments appearing after `response.completed` were not filtered
from the merged output, causing them to appear in rendered transcripts.

## Fix

- `_event_key` now uses `event_id` for all event types, including `tool_output`.
- Post-completion `debug_only` fragments are now dropped during merge.
- `incident_summary.py` now counts from merged events directly instead of
  rendered lines.

## Verification

Run the following to confirm the fix:

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
