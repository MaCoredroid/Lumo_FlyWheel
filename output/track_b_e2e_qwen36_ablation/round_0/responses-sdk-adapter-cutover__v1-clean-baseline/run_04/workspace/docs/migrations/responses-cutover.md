# Responses Cutover

## What Changed

The incident-handoff wire path was migrated from the legacy `chat_completions`
messages API to the `responses` API.  Configuration in `config/runtime.toml`
was updated accordingly:

```toml
wire_api = "responses"
transcript_mode = "responses_events"
```

Client constants (`WIRE_API`, `LEGACY_WRAPPER`) and `extract_response_items()`
were updated to read the Responses output structure (`response["output"]`)
instead of the nested `choices[0].message.content` path.

## Event Ordering

The adapter preserves the original ordering of events as returned by the
Responses API.  Each item in the response output is appended sequentially to
the event list:

- `assistant_text` — rendered text from the model
- `tool_call` — a tool invocation with `call_id`, `name`, and `arguments`
- `tool_result` — the result of a prior tool call, keyed by the same `call_id`

Reordering events (e.g., placing a `tool_result` before its matching
`tool_call`) would break downstream replay and rendering.  Consumers must
process events in the order delivered.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`.  The corresponding `tool_result`
event references the same `call_id` so that results can be correlated back to
their originating invocation.  The replay serialization (`replay.py`) now
preserves `call_id` in both serialization and deserialization for both
`tool_call` and `tool_result` events, ensuring round-trip fidelity.

## Replay

Replay remains event-sourced: `replay_from_serialized()` reconstructs the
original event list from pipe-delimited serialized lines.  State is **not**
rebuilt from rendered transcript text — the serialized event format is the
single source of truth.
