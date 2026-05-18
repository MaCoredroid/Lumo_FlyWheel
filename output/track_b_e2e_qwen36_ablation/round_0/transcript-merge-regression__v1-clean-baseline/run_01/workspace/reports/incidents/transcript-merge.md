# Transcript Merge Incident Note

## Root cause

The reducer in `replay/merge.py` used `(role, tool_name)` as the identity key
for `tool_output` events instead of `event_id`, so two distinct tool outputs
from the same tool were incorrectly collapsed into one block.

Additionally, debug-only fragments that arrived after `response.completed` were
tagged with `after_completion` but still emitted to the render pipeline.

The incident summary counted tool blocks from rendered lines instead of from
the merged event list, propagating the merge bug into reports.

## Fix

- `_event_key` now always returns `event_id` (or a `kind:sequence` fallback),
  giving every event a stable identity independent of role or tool name.
- `merge_records` skips `debug_only` events that appear after
  `response.completed`, so they no longer reach render or summary.
- `summarize_events` counts directly from merged events (tool_output and
  assistant kinds) instead of parsing rendered lines.

## Verification

`python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`
