# Responses Cutover

Migrated from the legacy chat-wrapper path to Responses event semantics.

## Config Changes

- `config/runtime.toml` — `wire_api` set to `"responses"`, `transcript_mode` set to `"responses_events"`.
- `src/incident_handoff/client.py` — wire config now returns Responses values; `extract_response_items` reads `response["output"]` instead of the legacy `choices[0].message.content`.

## Event Ordering

Events are emitted in strict chronological order as returned by the Responses API. The order within a turn is:

1. `assistant_text` — model-generated text segments.
2. `tool_call` — the model requests a tool invocation (includes `call_id`).
3. `tool_result` — the resolved result for a prior tool call (references the same `call_id`).

This ordering is preserved during serialization, deserialization, and rendering so that replay reconstructs the original turn faithfully.

## Tool-Result Correlation

Every `tool_call` event carries a unique `call_id`. The corresponding `tool_result` event references the same `call_id`, establishing a one-to-one correlation. Replay (`serialize_events` / `replay_from_serialized`) preserves `call_id` in both directions — serialization includes the field in the pipe-delimited format, and deserialization restores it on the reconstructed event dict. This ensures tool-result correlation survives round-trips through serialized storage.

## Replay

Replay remains event-sourced: state is reconstructed by replaying the serialized event stream, not by parsing rendered transcript text. See `src/incident_handoff/replay.py`.
