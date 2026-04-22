# Responses Cutover

This workspace now uses the Responses wire path and stores transcripts as
ordered Responses events.

- Set `wire_api = "responses"` and `transcript_mode = "responses_events"` in
  `config/runtime.toml`.
- Extract response items from the Responses `output` array instead of the legacy
  chat wrapper at `choices[0].message.content`.
- Preserve event ordering exactly as emitted. Assistant text blocks, tool calls,
  and tool results must remain in source order so replay reflects the original
  turn chronology.
- Preserve tool-result correlation with `call_id` on both `tool_call` and
  `tool_result` events. Rendering may be human-readable, but replay must stay
  event-sourced and consume the stored event stream rather than rebuilding state
  from rendered transcript text.
