# Responses Cutover

Migrated from the legacy chat-completions wrapper to the Responses API wire path
with event-sourced transcript mode.

## Wire Path

- `config/runtime.toml` now sets `wire_api = "responses"` and `transcript_mode = "event_sourced"`.
- `client.py` returns `event_sourced` transcript config and extracts items from `response["output"]` instead of `response["choices"][0]["message"]["content"]`.

## Event Ordering

Events are preserved in the exact order emitted by the Responses API:

1. `assistant_text` events appear first (model text output).
2. `tool_call` events follow, each carrying a `call_id` that uniquely identifies the invocation.
3. `tool_result` events appear after their corresponding `tool_call`, matched by `call_id`.

This ordering guarantees that replay reproduces the same sequential conversation flow without rebuilding state from rendered transcript text.

## Tool-Result Correlation

Every `tool_call` event carries a `call_id`. The corresponding `tool_result` event
references the same `call_id`, enabling one-to-one correlation between invocation
and outcome. Both `serialize_events` and `replay_from_serialized` preserve
`call_id` through the serialization round-trip, and `render_transcript` includes
`call_id` in the rendered output so that tool results remain attributable.

## Replay

Replay is event-sourced: it reconstructs state by re-applying the original event
sequence rather than parsing rendered transcript text. This avoids information loss
and keeps `call_id` correlation intact across serialization boundaries.
