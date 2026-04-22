# Responses Cutover

This workspace now runs on the Responses wire path and stores transcripts as ordered
Responses events.

- Set `config/runtime.toml` to `wire_api = "responses"` and
  `transcript_mode = "responses_events"`.
- Consume `response["output"]` items directly instead of rebuilding a legacy
  assistant message wrapper.
- Preserve event ordering exactly as emitted. For `message` items, flatten
  `content[*].type == "output_text"` blocks in order before moving to later
  tool events.
- Preserve tool-result correlation with the original `call_id` on both
  `tool_call` and `tool_result` events.
- Keep replay event-sourced by serializing the event stream itself. Do not
  rebuild state from rendered transcript text.
