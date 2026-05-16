# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are processed in strict sequence order as emitted by the Responses API.
- Each event carries a monotonically increasing sequence number.
- Replay must preserve original event order; do not reorder or deduplicate events.
- Out-of-order events are rejected during replay validation.

## Tool-Result Correlation

- Tool results are correlated to tool calls via `call_id`.
- Each `tool_call` event emits a unique `call_id`.
- The corresponding `tool_result` event references the same `call_id`.
- Correlation is enforced during replay: unmatched `call_id`s cause replay failure.

## Replay Semantics

- Replay is event-sourced; state is reconstructed from the event stream.
- Do not rebuild state from rendered transcript text.
- The transcript stores raw events, not rendered output.
