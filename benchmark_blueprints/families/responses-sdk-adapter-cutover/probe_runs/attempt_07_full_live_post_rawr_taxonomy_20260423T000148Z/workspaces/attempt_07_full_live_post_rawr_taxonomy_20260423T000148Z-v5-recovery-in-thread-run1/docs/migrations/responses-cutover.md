# Responses Cutover

This workspace is cut over to the Responses wire path and `responses_events`
transcript mode.

## Required runtime settings

- Set `wire_api = "responses"` in `config/runtime.toml`.
- Set `transcript_mode = "responses_events"` in `config/runtime.toml`.
- Read response items from the Responses `output` event list instead of the
  legacy chat-wrapper message envelope.

## Event ordering

- Preserve the original event stream as events, not as rendered transcript text.
- When a Responses item includes `sequence`, process events in ascending
  sequence order.
- If `sequence` is absent, preserve source order exactly as received.
- Expand message content blocks in the order they appear so multi-block
  assistant output remains replayable.

## Tool-result correlation

- Preserve every tool call `call_id`.
- Preserve every tool result `call_id`.
- Keep tool calls and tool results as separate event records; do not merge them
  into rendered transcript lines.
- Replay must preserve tool-result correlation from event data, not from text
  matching in rendered transcripts.

## Replay contract

- Replay stays event-sourced.
- Serialize and deserialize the canonical event records directly.
- Do not rebuild state from rendered transcript text.
- Unknown future Responses event types should be preserved as opaque events so
  ordering is not lost during replay.
