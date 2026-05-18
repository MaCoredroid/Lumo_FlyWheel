# Responses Cutover

Switched from the legacy `chat_completions` wrapper to the `responses` wire API with `event_stream` transcript mode.

## Event Ordering

Events arrive in strict arrival order within the event stream. Consumers must preserve this ordering when normalizing response items — the sequence of `assistant_text`, `tool_call`, and `tool_result` events in the output list must match the order produced by the Responses wire. Reordering events (for example, grouping all tool results after all tool calls) will break downstream replay and rendering.

## Tool-Result Correlation

Each `tool_result` event carries a `call_id` that references the originating `tool_call`. This correlation is required to reconstruct the interleaved tool-use flow during replay:

- `tool_call` events emit a `call_id` (e.g., `call-1`).
- The corresponding `tool_result` event carries the same `call_id`.
- Serialization (`serialize_events`) and deserialization (`replay_from_serialized`) must round-trip `call_id` without loss.
- Rendering (`render_transcript`) includes `call_id` in tool-result output so the correlation is visible in the transcript.

## Replay

Replay remains event-sourced: the serialized event log is the source of truth. State is reconstructed from the event log, not from rendered transcript text. The `call_id` field is preserved through the serialize/deserialize cycle to maintain tool-call-to-tool-result linkage.
