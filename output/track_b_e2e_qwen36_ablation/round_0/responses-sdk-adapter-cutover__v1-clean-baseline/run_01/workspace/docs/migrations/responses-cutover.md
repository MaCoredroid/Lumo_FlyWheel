# Responses Cutover

This workspace migrated from the legacy `chat_completions` wire path to the
Responses event semantics (`wire_api = "responses"`, `transcript_mode = "events"`).

## Event Ordering

Events are emitted and replayed in strict arrival order within each turn.
The serializer preserves the original sequence, so `assistant_text`,
`tool_call`, and `tool_result` events appear in the same order as
received from the Responses wire.  Never reorder events during
serialization or replay—doing so breaks causality for downstream
consumers.

## Tool-Result Correlation

Every `tool_call` event carries a `call_id`; the matching `tool_result`
event references the same `call_id`.  This correlation is preserved
through serialization, deserialization, and rendering:

- `serialize_events` writes `call_id` as the second pipe-delimited field
  for both `tool_call` and `tool_result`.
- `replay_from_serialized` reads that field back so the round-tripped
  event list is structurally identical to the input.
- `render_transcript` includes the `call_id` inline
  (e.g., `tool_result(call-1): oncall-primary`) so rendered output
  remains inspectable.

## Replay — Event-Sourced Only

Replay reconstructs state by replaying the raw event stream from the
transcript store.  It does **not** rebuild state by parsing rendered
transcript text.  The serialization format (`kind|call_id|...`) is the
source of truth; rendered text is an output side effect only.

## Adapter Changes

`normalize_response_items` now handles both Responses-native types
(`message`, `tool_call`, `tool_result`) and legacy alias types
(`function_call`, `function_call_output`) by mapping the appropriate
keys (`id`/`tool_name`/`tool_call_id`) to the unified event schema.
Message content is extracted from nested `output_text` blocks rather
than a flat string, matching the Responses wire format.
