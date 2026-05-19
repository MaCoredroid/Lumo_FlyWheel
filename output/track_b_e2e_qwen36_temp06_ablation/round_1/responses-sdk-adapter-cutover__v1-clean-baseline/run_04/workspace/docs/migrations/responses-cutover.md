# Responses Cutover

Migrated from the legacy `chat_completions` chat-wrapper path to the Responses API wire path with `responses_events` transcript mode.

## Wire Path

- `config/runtime.toml` now sets `wire_api = "responses"` and `transcript_mode = "responses_events"`.
- `client.py` reads from `response["output"]` instead of `response["choices"][0]["message"]["content"]`.
- `adapter.py` extracts text from the Responses content array (`output_text` parts) rather than a flat string.

## Event Ordering

Events are delivered and replayed in the order they appear in the Responses output array. The adapter preserves this ordering when normalising items into internal event kinds (`assistant_text`, `tool_call`, `tool_result`). Replay serialization and deserialization maintain the original sequence — state is reconstructed from the event stream, not from rendered transcript text.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`. The corresponding `tool_result` event references the same `call_id`. The replay module now preserves `call_id` through the serialize/deserialize round-trip so that tool results remain correlated with their originating calls. This ensures that multi-turn tool sequences (call → result → call → result) are faithfully reconstructed during replay.
