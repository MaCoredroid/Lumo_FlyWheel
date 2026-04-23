# Responses Cutover

This workspace now targets the Responses wire path and stores transcripts as
ordered Responses events rather than legacy wrapped messages.

## Required config

- Set `wire_api = "responses"`.
- Set `transcript_mode = "responses_events"`.

## Event ordering

- Normalize replay input from the Responses item stream, not from rendered
  transcript text.
- Respect `sequence` when present so reordered chunks are restored to original
  order before replay or rendering.
- Keep multi-block assistant output in block order when turning message items
  into transcript events.

## Tool-result correlation

- Preserve each tool call `call_id` on both the tool call event and the
  matching tool result event.
- Render tool results with their `call_id` so audits can correlate a result
  back to the originating tool invocation.

## Replay constraints

- Keep replay event-sourced by serializing structured events directly.
- Preserve unknown future Responses event payloads, or ignore them explicitly,
  without disturbing neighboring event order.
- Do not rebuild replay state from rendered transcript text.
