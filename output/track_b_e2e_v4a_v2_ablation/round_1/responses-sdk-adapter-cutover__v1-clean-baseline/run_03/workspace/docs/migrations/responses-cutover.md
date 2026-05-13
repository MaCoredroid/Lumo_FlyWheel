# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are stored and replayed in strict sequential order as they occurred.
- The transcript mode is set to `events` to preserve the original event stream.
- Do not reorder events during serialization or replay; maintain the original index-based ordering.
- When replaying, process events in the exact order they appear in the transcript.

## Tool-Result Correlation

- Each `tool_result` event must be correlated with its corresponding `tool_call` event via `call_id`.
- The `call_id` field is the primary key for linking tool invocations to their results.
- Ensure that tool results appear after their associated tool calls in the event stream.
- When reconstructing state, match tool results to calls using the `call_id` field.

## Replay Behavior

- Replay is event-sourced: events are processed directly from the transcript.
- Do not rebuild state from rendered transcript text; use the raw event stream.
- The `transcript_mode = "events"` setting ensures event-level fidelity during replay.
