# Transcript Merge Incident Note

## Resolution

## Root cause

`_event_key` in `replay/merge.py` grouped `tool_output` events by
`(role, tool_name)` instead of `event_id`, causing distinct tool outputs
with the same name to be incorrectly merged into a single block.
Additionally, post-completion `debug_only` fragments were retained in the
merged output and rendered as regular tool lines.

## Fix applied

- `_event_key` now returns `event_id` for `tool_output` events, giving each
  tool output stable individual identity.
- `merge_records` filters out events marked both `after_completion` and
  `debug_only` before returning the merged list.
- `incident_summary.py` now counts tool-output blocks directly from merged
  events (`count_source: merged_events`) instead of from rendered lines.

## Verification

All tests pass:

- `test_merge` — distinct tool outputs remain separate after merge
- `test_render` — debug-only post-completion fragments are not rendered
- `test_incident_summary` — summary counts merged tool blocks correctly
