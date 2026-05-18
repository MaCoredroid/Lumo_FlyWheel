# Responses Cutover

## Summary

Migrated from the legacy `chat_completions` wire path with `legacy_messages` transcript mode to the Responses API wire path with `event_sourced` transcript mode.

## Config Changes

- `config/runtime.toml`: `wire_api` set to `"responses"`, `transcript_mode` set to `"event_sourced"`.
- `client.py`: `WIRE_API` constant and `request_wire_config()` updated accordingly; `LEGACY_WRAPPER` disabled.

## Event Ordering

Events are delivered in strict wire order. The adapter preserves the original sequence of `message`, `tool_call`, and `tool_result` items as they appear in the Responses output array. Replay serialization and deserialization maintain this ordering — events are written and read back line-by-line in the same order.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`. The corresponding `tool_result` event references the same `call_id`. The replay module now preserves `call_id` through the serialize/deserialize round-trip so tool calls and their results remain correlated after replay. The render module includes `call_id` in tool-result output for traceability.

## Replay

Replay remains event-sourced: state is reconstructed by replaying the serialized event log, not by parsing rendered transcript text.
