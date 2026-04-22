# Responses Cutover

Use the Responses wire path and persist the raw event stream as the transcript source.

- Request the `responses` wire API and store transcripts in `responses_events` mode.
- Normalize ordered response items into ordered internal events without collapsing them into legacy message wrappers.
- Preserve tool-result correlation with the original `call_id` on both `tool_call` and `tool_result` events.
- Keep replay event-sourced by reading serialized events directly; do not rebuild state from rendered transcript text.
- Maintain event ordering exactly as emitted so assistant text, tool calls, and tool results replay in the same sequence.
