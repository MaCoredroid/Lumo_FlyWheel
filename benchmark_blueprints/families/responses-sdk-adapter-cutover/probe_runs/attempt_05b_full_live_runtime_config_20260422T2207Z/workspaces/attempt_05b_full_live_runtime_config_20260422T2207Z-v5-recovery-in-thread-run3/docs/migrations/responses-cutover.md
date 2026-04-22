# Responses Cutover

This workspace now uses the Responses wire path and event transcript mode:

- `wire_api = "responses"`
- `transcript_mode = "responses_events"`

Cutover rules:

- Consume `response.output` items as the source of truth instead of the legacy chat wrapper.
- Preserve event ordering from the Responses event stream. When event `sequence` values are present, normalize by sequence order rather than input file order.
- Preserve tool-result correlation by carrying `call_id` on both `tool_call` and `tool_result` events through normalization, replay serialization, and rendering.
- Keep replay event-sourced. Serialize and replay the normalized event objects directly; do not rebuild state from rendered transcript text.
- Pass through unknown or future Responses item types as raw event items so replay remains forward-compatible without losing ordering.
