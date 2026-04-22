# Responses Cutover

This workspace now uses the Responses wire path and stores transcripts as ordered
Responses events.

- Set `config/runtime.toml` to `wire_api = "responses"` and
  `transcript_mode = "responses_events"`.
- Normalize the API payload from `response["output"]` into event records rather
  than reading legacy chat-wrapper message text.
- Preserve event ordering exactly as emitted. Replay must consume the serialized
  event stream in sequence and must not rebuild state from rendered transcript
  text.
- Preserve tool-result correlation with `call_id` on both `tool_call` and
  `tool_result` events. Rendering may be human-readable, but replay must stay
  event-sourced and correlation-aware.
- Message content should be extracted from Responses content blocks such as
  `output_text` blocks and emitted as `assistant_text` events in-order with tool
  events.
