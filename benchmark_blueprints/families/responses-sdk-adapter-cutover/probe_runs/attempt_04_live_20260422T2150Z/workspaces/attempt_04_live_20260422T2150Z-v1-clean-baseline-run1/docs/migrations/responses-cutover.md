# Responses Cutover

This workspace now uses the Responses wire path and persists transcripts as ordered
Responses events instead of legacy wrapped messages.

## Required config

Set `.codex/config.toml` to:

```toml
wire_api = "responses"
transcript_mode = "responses_events"
```

## Event ordering

Preserve the event stream exactly as emitted. Assistant text, tool calls, and
tool results must remain in source order so downstream replay sees the same
interleaving that the model produced.

## Tool-result correlation

Treat `call_id` as the stable join key between each `tool_call` and its
matching `tool_result`. Do not drop `call_id` during normalization, replay, or
rendering. Rendered transcripts may include the identifier for readability, but
correlation must come from the structured event payloads.

## Replay contract

Replay stays event-sourced. Serialize and reload the structured events
themselves, preserving ordering and all tool metadata, rather than rebuilding
state from rendered transcript text.
