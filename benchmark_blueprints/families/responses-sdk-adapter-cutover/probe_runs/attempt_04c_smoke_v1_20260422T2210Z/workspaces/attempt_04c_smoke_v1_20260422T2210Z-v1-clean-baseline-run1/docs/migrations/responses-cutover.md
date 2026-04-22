# Responses Cutover

This workspace now uses the Responses wire path and stores transcripts as ordered
Responses events.

## Required configuration

Set `.codex/config.toml` to:

```toml
wire_api = "responses"
transcript_mode = "responses_events"
```

## Event handling rules

- Consume the Responses item list directly from the wire payload. Do not unwrap
  through the legacy chat-completions message shape.
- Preserve event ordering exactly as emitted. Message blocks, tool calls, and
  tool results must remain in wire order.
- Correlate each `tool_result` to its originating `tool_call` with `call_id`.
  Rendering may summarize events for humans, but replay must keep the original
  `call_id` values intact.
- Keep replay event-sourced. Rehydrate from serialized events, not from rendered
  transcript text.

## Migration notes

- Normalize message content from Responses output blocks into assistant text
  events without collapsing surrounding tool events.
- Serialize replay data from the event objects themselves so arguments, outputs,
  ordering, and `call_id` correlation round-trip losslessly.
- Rendered transcripts are for inspection only and must not become the replay
  source of truth.
