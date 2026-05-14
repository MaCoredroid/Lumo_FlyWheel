# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events must be processed in strict sequential order as emitted by the Responses API.
- The event stream order defines causal relationships between assistant messages, tool calls, and tool results.
- Do not reorder or batch events; preserve the original sequence for accurate replay.

## Tool-Result Correlation

- Tool results are correlated to their corresponding tool calls via the `call_id` field.
- Each `tool_call` event produces a matching `tool_result` event with the same `call_id`.
- Maintain this pairing during replay.
- Do not attempt to infer tool call results from rendered transcript text; use the event-sourced `tool_result` events directly.

## Replay Semantics

- Replay is event-sourced: reconstruct state by replaying the ordered event stream.
- Do not rebuild state from rendered transcript text; rely on the structured event data.
