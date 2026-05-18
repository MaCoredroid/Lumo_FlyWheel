# Responses Cutover

Switched from the legacy `chat_completions` wrapper to the Responses wire API
(`wire_api = "responses"`) with event-sourced transcript mode
(`transcript_mode = "events"`).

## Event Ordering

Events are emitted in strict chronological order as they appear in the
Responses output array: `assistant_text`, `tool_call`, `tool_result`. The
replay module preserves this ordering through serialization — `serialize_events`
writes events sequentially and `replay_from_serialized` reconstructs them in
the same order. Never reorder events during replay.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`. The corresponding `tool_result`
event references the same `call_id`, establishing a one-to-one correlation
between invocation and outcome. Both `serialize_events` and
`replay_from_serialized` preserve `call_id` fields so that tool results can be
matched to their originating calls after deserialization.

## Replay

Replay is event-sourced: state is reconstructed from the serialized event
stream, not from rendered transcript text. The `render_transcript` function
produces a human-readable view only and must not be used as a source of truth
for replay.
