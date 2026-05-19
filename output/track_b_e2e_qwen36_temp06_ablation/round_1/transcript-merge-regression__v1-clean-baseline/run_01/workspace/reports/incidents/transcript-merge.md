# Transcript Merge Incident Note

## Resolution

## What was fixed

Three bugs in the transcript merge pipeline were identified and repaired:

1. **`replay/merge.py` – `_event_key`**: Tool-output events were keyed by `(role, tool_name)` instead of `event_id`, causing distinct tool calls with the same name to collapse into a single merged record. Fixed to use `event_id` for stable identity.

2. **`replay/merge.py` – post-completion debug fragments**: Debug-only fragments arriving after `response.completed` were retained in the merged output and rendered as visible tool blocks. Fixed to evict them during merge so they never reach the renderer.

3. **`replay/incident_summary.py` – `summarize_events`**: The summary counted rendered lines (`rendered_lines`) instead of merged event blocks. Fixed to count directly from merged events and report `count_source` as `merged_events`.

## Files updated: `replay/merge.py`, `replay/incident_summary.py`.

## Files updated

- `replay/merge.py` – stable event identity + debug-fragment eviction
- `replay/render.py` – keep honest (no extra filtering needed; merge now handles it)
- `replay/incident_summary.py` – count merged blocks, not rendered lines

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
