# Responses Cutover

Migrate from the legacy message wrapper to Responses event semantics.

## Event Ordering

Events are now streamed in strict chronological order. Each event carries a monotonically increasing sequence number used for replay. Do not reorder events during transcript processing.

## Tool-Result Correlation

Tool calls and their results are correlated via `call_id`. When replaying:
- Emit `tool_call` events with their `call_id`, `name`, and `arguments`.
- Emit `tool_result` events with the matching `call_id` and `output`.
- Maintain the original sequence: tool results must follow their corresponding tool calls.

## Replay

Replay remains event-sourced. Do not rebuild state from rendered transcript text; instead, re-emit the original event stream in order.
