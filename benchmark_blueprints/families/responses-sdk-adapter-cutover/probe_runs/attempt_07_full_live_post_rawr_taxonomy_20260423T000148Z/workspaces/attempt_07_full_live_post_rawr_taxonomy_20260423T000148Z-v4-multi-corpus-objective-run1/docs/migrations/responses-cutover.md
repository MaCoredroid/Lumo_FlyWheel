# Responses Cutover

This workspace now uses the Responses wire path and persists Responses event transcripts directly.

- Set `wire_api = "responses"` in `config/runtime.toml`.
- Set `transcript_mode = "responses_events"` in `config/runtime.toml`.
- Read response items from the Responses `output` array instead of the legacy chat-wrapper `choices[].message.content` path.
- Preserve event ordering by replaying the recorded event stream in sequence order. If an event payload includes a `sequence` field, normalize against that field rather than the file's incidental object order.
- Preserve tool-result correlation with `call_id` on both `tool_call` and `tool_result` events. Rendered transcripts are for display only and must not become the source of truth for correlation.
- Keep replay event-sourced. Serialize and restore the structured event objects directly; do not rebuild state from rendered transcript text.
