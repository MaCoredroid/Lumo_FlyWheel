# Responses Cutover

This workspace is cut over to the Responses wire path and stores transcripts as
ordered Responses events.

## Required runtime settings

- Set `config/runtime.toml` to `wire_api = "responses"`.
- Set `config/runtime.toml` to `transcript_mode = "responses_events"`.

## Event normalization

- Consume `response["output"]` items directly instead of rebuilding a legacy
  chat-wrapper message payload.
- Normalize events in recorded order. When a `sequence` field is present, use it
  to restore the original event order before flattening content.
- For `message` items, emit `content[*].type == "output_text"` blocks in the
  same order they were recorded.

## Ordering and tool correlation

- Preserve event ordering exactly as recorded so replay stays event-sourced.
- Preserve tool-result correlation with the original `call_id` on both
  `tool_call` and `tool_result` events.
- Do not infer tool-result linkage from rendered transcript text or positional
  adjacency.

## Replay contract

- Serialize and deserialize the event records themselves.
- Keep replay event-sourced; do not rebuild state from rendered transcript text.
