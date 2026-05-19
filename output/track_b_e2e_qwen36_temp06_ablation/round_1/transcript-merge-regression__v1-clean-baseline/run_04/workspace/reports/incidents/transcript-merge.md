# Transcript Merge Incident Note

## Root cause

The reducer in `replay/merge.py` used `(role, tool_name)` as the grouping key for
`tool_output` events, which collapsed distinct tool calls sharing the same name
(e.g. two `fetch_weather` calls) into a single merged block.  Additionally,
debug-only fragments arriving after `response.completed` were marked
`after_completion` but never excluded from rendering, so they appeared as
duplicate tool lines in the final output.

## Fixes applied

- `_event_key` now uses `event_id` for all event kinds (including `tool_output`),
  giving each tool call a stable, unique identity.
- `render_events` skips events flagged with both `after_completion` and
  `debug_only`, so post-completion debug fragments no longer leak into the
  rendered transcript.
- `summarize_events` counts tool-output blocks directly from merged events
  (`count_source: merged_events`) instead of parsing rendered lines.

## Status

Reducer bug resolved.  Render filtering and incident summary are now correct.
