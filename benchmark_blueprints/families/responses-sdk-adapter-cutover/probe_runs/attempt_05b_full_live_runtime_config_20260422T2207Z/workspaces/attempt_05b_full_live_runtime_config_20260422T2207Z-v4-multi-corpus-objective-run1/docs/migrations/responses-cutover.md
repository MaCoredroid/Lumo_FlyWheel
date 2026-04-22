# Responses Cutover

This workspace now uses the Responses wire path and stores transcripts as
Responses events instead of legacy rendered messages.

## Required runtime settings

- Set `wire_api = "responses"` in `config/runtime.toml`.
- Set `transcript_mode = "responses_events"` in `config/runtime.toml`.
- Disable any legacy chat-wrapper compatibility path for request extraction.

## Event ordering

Replay must consume structured events in event order. When transcript items carry
an explicit sequence field, that sequence is the source of truth for ordering.
Do not depend on incidental file order or on the rendered transcript text.

## Tool-result correlation

Tool calls and tool results must remain correlated by `call_id`.

- Preserve `call_id` on every normalized `tool_call` event.
- Preserve `call_id` on every normalized `tool_result` event.
- Rendered transcripts may display `call_id` for debugging, but replay must read
  the structured event stream rather than reconstructing state from rendered text.

## Replay rule

Keep replay event-sourced. Serialize and reload the event objects themselves so
ordering and tool-result correlation survive round trips without relying on any
lossy text rendering layer.
