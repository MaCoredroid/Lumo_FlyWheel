# Responses Cutover

## Summary

Migrated the incident-handoff wire path from the legacy `chat_completions` chat-wrapper to the `responses` API with `event_sourced` transcript mode.

## Config Changes

- `config/runtime.toml`: `wire_api` set to `"responses"`, `transcript_mode` set to `"event_sourced"`.
- `src/incident_handoff/client.py`: `WIRE_API` and `request_wire_config()` updated to match. `extract_response_items` now reads from `response["output"]` (Responses event list) instead of `response["choices"][0]["message"]["content"]`.

## Event Ordering

Events are delivered in strict chronological order as emitted by the Responses API. The adapter (`src/incident_handoff/adapter.py`) preserves this order when normalizing into internal event kinds (`assistant_text`, `tool_call`, `tool_result`). Consumers must process events sequentially; reordering or parallelizing event ingestion is not supported.

## Tool-Result Correlation

Each tool call is identified by a unique `call_id`. The corresponding tool result carries the same `call_id` so that consumers can correlate results back to their originating call. This correlation is preserved through the full pipeline:

1. **Adapter** — normalizes `tool_call` / `tool_result` (and legacy `function_call` / `function_call_output`) events, keeping `call_id` intact.
2. **Replay** — `serialize_events` and `replay_from_serialized` round-trip `call_id` for both `tool_call` and `tool_result` events so that replayed event streams maintain the same call-to-result mapping as the original stream.

## Replay

Replay remains event-sourced: the serialized event log is the source of truth. State is never rebuilt from rendered transcript text.

## Legacy Type Aliases

The adapter handles legacy `function_call` and `function_call_output` types (found in older transcript fixtures) by mapping them to the canonical field names (`id` → `call_id`, `tool_name` → `name`, `tool_call_id` → `call_id`).
