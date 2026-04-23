# Responses Cutover

Use the Responses wire path and persist ordered response events directly.

- Set `wire_api = "responses"` and `transcript_mode = "responses_events"`.
- Normalize `message` items by walking each content block in order and emitting event records from the raw Responses payload.
- Preserve tool-result correlation by carrying the same `call_id` on both the `tool_call` and `tool_result` events.
- Keep replay event-sourced: serialize and replay the raw ordered events, not rendered transcript text.
- Treat rendered transcripts as a view only. They may include `call_id` for operator readability, but replay must continue to rebuild from the stored event stream.
