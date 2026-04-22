# Responses Cutover

This workspace now uses the Responses wire path and stores transcripts in
Responses event mode.

## Runtime Configuration

- Set `wire_api = "responses"` in `config/runtime.toml`.
- Set `transcript_mode = "responses_events"` in `config/runtime.toml`.
- Read top-level response items from `response.output` and only fall back to a
  nested response envelope when the caller still wraps the SDK payload.

## Event Ordering

- Normalize response items in event order, not arrival order.
- When a response item includes `sequence`, sort by `sequence` before emitting
  normalized events.
- Within a single message item, preserve the original order of the message
  content blocks when flattening `output_text` entries into transcript events.
- Keep replay event-sourced: replay must round-trip structured events rather
  than trying to rebuild state from rendered transcript text.

## Tool Result Correlation

- Preserve each tool call's `call_id` on both the `tool_call` event and its
  matching `tool_result` event.
- Rendered transcript text may display `call_id` for debugging, but replay and
  state recovery must rely on the structured event stream.
- Unknown future Responses item types should pass through as structured events
  so event ordering stays intact and downstream replayers can decide how to
  handle them without losing correlation context.
