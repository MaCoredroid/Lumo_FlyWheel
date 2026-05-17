# Transcript Merge Incident Note

## Resolution

## Root Cause

The reducer in `replay/merge.py` had two bugs:

1. **Unstable event identity for tool outputs**: The `_event_key` function grouped tool outputs by `(role, tool_name)` instead of using `event_id`, causing distinct tool outputs with the same tool name to be incorrectly merged.

2. **Debug-only fragments after completion were not filtered out, causing them to appear in rendered output.

## Fix Applied

- `_event_key` now uses `event_id` for stable identity (falling back to `kind:sequence` only when `event_id` is absent).
- `merge_records` now skips `debug_only` fragments that appear after `response.completed`.
- `incident_summary.py` now counts from merged events directly instead of counting rendered lines.

## Verification

Run the following to verify the fix:

```bash
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
