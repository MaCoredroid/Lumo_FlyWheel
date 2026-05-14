# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are processed in strict arrival-order; preserve sequence IDs from the wire.
- Replay uses event-sourced transcript; do not reorder or deduce ordering from rendered text.
- Maintain causal links between tool calls and their results via event IDs.

## Tool-Result Correlation

- Each tool call event carries a unique call_id; results reference the same call_id.
- Correlate tool outputs by matching call_id in the event stream.
- Do not infer tool results from rendered transcript text; use event payloads.

## Wire Path

- wire_api = "responses"
- transcript_mode = "responses_events"
