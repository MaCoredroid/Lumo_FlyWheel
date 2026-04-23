# Responses Cutover

This workspace now uses the Responses wire path and stores transcript state as
Responses event items instead of the legacy chat-wrapper message shape.

## Runtime config

- Set `wire_api = "responses"` in `config/runtime.toml`.
- Set `transcript_mode = "response_events"` in `config/runtime.toml`.
- Read response items from the Responses `output` array, not from
  `choices[0].message.content`.

## Event ordering

- Treat the Responses item stream as the source of truth for replay.
- Preserve item order exactly as emitted, using the event `sequence` field when
  present so reordered payload arrays are normalized back into event order.
- Split message blocks into assistant text events in their original block order.

## Tool-result correlation

- Preserve `call_id` on both `tool_call` and `tool_result` events.
- Render tool results with their `call_id` so the transcript keeps the tool
  result attached to the originating call.
- Keep replay event-sourced from serialized events; do not rebuild state from
  rendered transcript text.
