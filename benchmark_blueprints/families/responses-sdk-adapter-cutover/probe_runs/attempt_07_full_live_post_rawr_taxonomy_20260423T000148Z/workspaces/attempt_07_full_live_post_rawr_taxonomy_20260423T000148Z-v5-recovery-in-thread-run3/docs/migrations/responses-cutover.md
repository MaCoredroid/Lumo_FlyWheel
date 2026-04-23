# Responses Cutover

This workspace now uses the Responses wire path and stores transcripts in
event form rather than legacy message wrappers.

## Required runtime settings

- `wire_api = "responses"`
- `transcript_mode = "responses_events"`

## Event ordering

Preserve the Responses item order exactly as emitted. Normalize message items by
walking each message content block in sequence and emitting one event per
`output_text` block. Do not collapse the stream into rendered transcript text
and do not reorder tool activity around assistant text.

## Tool-result correlation

Treat `call_id` as the correlation key for tool activity. Both `tool_call` and
`tool_result` events must retain the same `call_id` through normalization,
serialization, replay, and human-readable rendering so replay can reconstruct
the event stream without guessing.

## Replay model

Replay stays event-sourced. Serialize the canonical event records directly and
replay from those records, not from a rendered transcript. Rendered transcript
output is for display only and must not become the source of truth for state
rebuilds.
