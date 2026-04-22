# Responses Cutover

Use the Responses wire path and persist the transcript as ordered response events.

## Required runtime settings

- `wire_api = "responses"`
- `transcript_mode = "responses_events"`

## Event ordering

Replay must consume the serialized event stream directly and preserve event order exactly as emitted.
Do not rebuild state from rendered transcript text or any other presentation layer.
Assistant message items should be expanded from Responses content blocks in block order so interleaved tool activity stays stable.

## Tool-result correlation

Every tool call and tool result must retain the original `call_id`.
Rendered transcripts should include the `call_id` on both `tool_call(...)` and `tool_result(...)` lines so correlation remains visible.
Serialized replay data must keep the full event payload, including `call_id`, instead of dropping fields during round-trip parsing.
