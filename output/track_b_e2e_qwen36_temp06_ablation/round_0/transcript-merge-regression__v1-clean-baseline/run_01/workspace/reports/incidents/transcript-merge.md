# Transcript Merge Incident Note

## Resolution

The reducer in `replay/merge.py` has been rewritten so that:

- Tool-output events use their `event_id` for stable identity, keeping distinct
  tool calls (even with the same `tool_name`) separate.
- Debug-only fragments that arrive after a `response.completed` event are
  dropped during merging rather than filtered at render time.

`replay/incident_summary.py` now counts tool-output blocks directly from the
merged event list instead of parsing rendered lines.

## Old guidance (no longer needed)

Operator render filtering is no longer required; the reducer correctly
deduplicates and prunes events. The incident summary counts merged tool-output
blocks rather than rendered tool lines.
