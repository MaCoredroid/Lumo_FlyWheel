# Responses Cutover

Migrated from the legacy `chat_completions` wrapper to the Responses API wire path
(`wire_api = "responses"`) and Responses event transcript mode
(`transcript_mode = "responses_events"`).

## Event Ordering

Events are consumed

- The event stream is now consumed as a flat Responses events (`message`, `tool_call`, `tool_result`)
  rather than nested chat-completion message blocks.
- Event ordering is preserved as-is from the wire: `assistant_text` → `tool_call` →
  `tool_result` sequences appear in the original order emitted by the Responses API.
- Replay remains event-sourced; state is never reconstructed from rendered transcript text.

## Tool-Result Correlation

- Every `tool_call` carries a `call_id`; its matching `tool_result` carries the same `call_id`.
- The `call_id` is preserved through serialization round-trips (`serialize_events` /
  `replay_from_serialized`) so that tool results can always be correlated back to their
  originating tool call.
- The adapter normalizes Responses `output_text` content blocks into the internal `assistant_text`
  events while keeping `call_id` fields intact on tool events.
