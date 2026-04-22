# Responses Cutover

Use the Responses wire path and persist the transcript as ordered Responses events.

- Set `wire_api = "responses"` and `transcript_mode = "responses_events"` in `config/runtime.toml`.
- Normalize the Responses item stream directly from the event payload. Do not read assistant state back out of rendered transcript text.
- Preserve event ordering exactly as emitted. Message blocks, tool calls, and tool results must remain in stream order during normalization, serialization, replay, and rendering.
- Preserve tool-result correlation with the original `call_id` on both `tool_call` and `tool_result` events.
- Keep replay event-sourced. Reconstruct replay state from the serialized event stream itself, not from human-readable transcript rendering.
