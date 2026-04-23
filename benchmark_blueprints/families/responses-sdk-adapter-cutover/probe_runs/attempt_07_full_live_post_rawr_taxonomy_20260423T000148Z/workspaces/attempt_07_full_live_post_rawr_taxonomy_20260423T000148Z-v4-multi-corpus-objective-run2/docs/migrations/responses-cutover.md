# Responses Cutover

This workspace now uses the Responses wire path and `responses_events` transcript mode.

Migration requirements:

- Read response items from the Responses `output` event stream instead of the legacy chat wrapper.
- Preserve event ordering exactly as emitted. Do not reorder message, tool call, or tool result events during normalization, replay, or rendering.
- Preserve tool-result correlation with the originating `call_id` on both `tool_call` and `tool_result` events.
- Keep replay event-sourced. Serialize and replay the event records themselves; do not rebuild state from rendered transcript text.
- Treat rendered transcripts as a view only. They are not the replay source of truth.
