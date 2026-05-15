# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are processed in strict arrival-order from the Responses wire path.
- Sequence numbers (if present) must be preserved during replay.
- Do not reorder events; maintain the original event stream order.

## Tool-Result Correlation

- Tool calls and their results are correlated via `call_id` (or equivalent event ID).
- When replaying, match tool-result events to their corresponding tool-call events using this ID.
- Preserve the association between tool-call and tool-result events in the transcript.

## Replay Semantics

- Replay is event-sourced: process the raw event stream directly.
- Do not rebuild state from rendered transcript text.
- Use the Responses event transcript mode to retain full event fidelity.
