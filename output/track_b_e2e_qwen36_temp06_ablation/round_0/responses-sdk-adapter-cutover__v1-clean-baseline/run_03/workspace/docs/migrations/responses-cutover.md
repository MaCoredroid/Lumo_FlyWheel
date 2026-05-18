# Responses Cutover

Migrated from the legacy chat-wrapper path to Responses event semantics.

## Config changes

- `config/runtime.toml` — `wire_api` set to `responses`, `transcript_mode` set to `responses_events`.
- `src/incident_handoff/client.py` — `WIRE_API` and `request_wire_config()` updated to the Responses wire path.

## Event ordering

Events are delivered in strict Responses order:

1. `message` (assistant text)
2. `tool_call` / `function_call`
3. `tool_result` / `function_call_output`

This ordering must be preserved during serialization, replay, and rendering. The `replay_from_serialized` function reconstructs the original event list in the same order produced by `serialize_events`.

## Tool-result correlation

Each `tool_result` event carries a `call_id` that references the preceding `tool_call` event. The adapter normalizes both Responses-native (`call_id`) and legacy alias (`tool_call_id`) fields to a single `call_id` key so downstream code can correlate tool calls with their results. This correlation is preserved through serialization and replay — `call_id` is never dropped.

## Replay

Replay remains event-sourced: the serialized event stream is the source of truth. State is **not** rebuilt from rendered transcript text.
