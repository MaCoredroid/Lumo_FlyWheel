# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are processed in strict sequential order as emitted by the Responses API.
- The event stream preserves causal ordering; do not reorder or batch events during replay.
- Each event carries a monotonically increasing sequence identifier for deterministic replay.
- Replay must consume events in the same order as original ingestion.

## Tool-Result Correlation

- Tool calls and their results are correlated via the `call_id` field on each event.
- A `tool_call` event is paired with a subsequent `tool_result` events sharing the same `call_id`.
- During replay, maintain the correlation map to ensure results are applied to the correct tool invocation.
- Do not attempt to correlate tool results by matching rendered transcript text; rely solely on event identifiers.

## Replay Semantics

- Replay is event-sourced: state is reconstructed by applying the event stream in order.
- Do not rebuild state from rendered transcript text; use the raw event payloads.
- The `responses_events` transcript mode ensures events are stored in their native format.
