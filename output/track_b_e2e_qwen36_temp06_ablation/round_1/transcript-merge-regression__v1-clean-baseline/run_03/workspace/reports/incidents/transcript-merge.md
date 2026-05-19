# Transcript Merge Incident Note

## Resolution

The reducer in `replay/merge.py` has been fixed:

- **Stable event identity** — `_event_key` now uses `event_id` for all event
  kinds (including `tool_output`), so distinct tool outputs no longer collapse
  into a single merged block.
- **Post-completion debug fragments** — events marked `debug_only` that arrive
  after a `response.completed` event are now dropped during merge, so they no
  longer leak into rendered output.

`replay/render.py` was already correct and requires no changes.

`replay/incident_summary.py` now counts directly from merged events
(`count_source: "merged_events"`) instead of parsing rendered lines.

## Verification

```
python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary
```
