# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Configuration Changes

- The runtime now uses the Responses wire path (`wire_api = "responses"`).
- Transcript mode is set to Responses events (`transcript_mode = "responses_events"`).

## Event Ordering

- Event ordering is preserved end-to-end. Each event carries a monotonically increasing sequence ID that must be respected during replay.
- Do not reorder, deduplicate, or skip events during transcript processing.
- Event sequence IDs are the source of truth for ordering; timestamps are informational only.

## Tool-Result Correlation

- Tool calls and their results are correlated via stable `call_id` values.
- Each `tool_call` event includes a unique `call_id`; the corresponding `tool_result` event references the same `call_id`.
- During replay, maintain a pending tool-call map keyed by `call_id` to match results to their originating calls.
- Do not infer tool outcomes from rendered transcript text; use only the structured `tool_result` events.

## Replay Semantics

- Replay is event-sourced: state is reconstructed by applying the event log in sequence.
- Do not rebuild state from rendered transcript text.
- Preserve the original event stream; do not normalize or transform event payloads during replay.
