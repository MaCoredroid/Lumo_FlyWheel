# Responses Cutover

Migrated from the legacy `chat_completions` wire path to the `responses` wire path.

## Wire Path

- `wire_api` is now `responses` (was `chat_completions`).
- `transcript_mode` is now `responses_events` (was `legacy_messages`).
- `client.extract_response_items` reads `response["output"]` instead of `response["choices"][0]["message"]["content"]`.

## Event Ordering

Events are delivered in strict arrival order from the Responses API. The adapter
preserves this order when normalizing response items — `message`, `tool_call`, and
`tool_result` events appear in the same sequence as emitted by the wire. Replay
serialization and deserialization maintain the original ordering; no re-sorting is applied.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`. The corresponding `tool_result` event
references the same `call_id`. This correlation is preserved through:

- **Adapter normalization** — `call_id` is copied verbatim from the wire payload.
- **Replay serialization** — `call_id` is the first field after the kind prefix
  in both `tool_call` and `tool_result` lines, so round-trips are lossless.
- **Replay deserialization** — `call_id` is restored from the serialized line so that
  downstream consumers can match results back to their originating calls.

## Replay

Replay remains event-sourced. State is reconstructed by deserializing the event log,
not by re-parsing rendered transcript text. This guarantees deterministic replay
regardless of rendering changes.
