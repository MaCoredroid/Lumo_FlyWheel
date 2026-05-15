# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are emitted in strict chronological order by the Responses API.
- Preserve the original event sequence during replay; do not reorder or deduplicate.
- Event timestamps and sequence IDs must be maintained for deterministic replay.

## Tool-Result Correlation

- Tool call events and their corresponding result events are correlated via a stable `call_id`.
- The `call_id` is generated at tool invocation and reused in the tool result event.
- Ensure tool results are matched to their calls by `call_id`, not by position or timing.
- Do not reconstruct execution flow by pairing tool calls with results using this correlation.

## Replay Semantics

- Replay is event-sourced: process the raw event stream directly.
- Do not rebuild state from rendered transcript text.
- Maintain event-sourced replay; do not rebuild state from rendered transcript text.
