# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events must be replayed in the exact order they were recorded.
- Preserve the sequence of all event types (message, tool_call, tool_result, etc.).
- Do not reorder or filter events during replay.

## Tool-Result Correlation

- Tool results must be correlated with their corresponding tool calls using the call ID.
- Maintain the bidirectional mapping between tool_call events and tool_result events.
- Ensure tool results are applied in the correct context of the conversation state.

## Replay Semantics

- Replay is event-sourced; do not rebuild state from rendered transcript text.
- Apply events incrementally to reconstruct conversation state.
