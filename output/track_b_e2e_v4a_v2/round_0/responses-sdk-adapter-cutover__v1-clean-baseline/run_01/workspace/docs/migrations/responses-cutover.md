# Responses Cutover

Migrate from legacy chat-wrapper path to Responses event semantics.

## Event Ordering

Events must be processed in strict chronological order by event index. Each event carries a monotonically increasing sequence number that defines its position in the transcript. Do not reorder events during replay; preserve the original sequence to maintain causal relationships between events.

## Tool-Result Correlation

Tool calls and their results are correlated by a unique call ID. When replaying events:
- Match `tool_call` events with their corresponding `tool_result` events using the call ID field.
- Preserve the pairing integrity; do not associate a tool result with an incorrect call.
- If a tool result is missing during replay, surface the orphaned call as an error.

## Replay Semantics

Replay remains event-sourced: rebuild state by applying events in order, not by parsing rendered transcript text. This ensures deterministic replay and preserves the full event history including intermediate states.
