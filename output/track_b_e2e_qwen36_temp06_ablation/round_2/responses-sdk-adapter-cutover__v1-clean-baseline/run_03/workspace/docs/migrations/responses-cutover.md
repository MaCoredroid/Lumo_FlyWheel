# Responses Cutover

Switched to Responses wire path and event-sourced transcript mode.

## What Changed

- **Wire API switched from `chat_completions` to `responses` (`config/runtime.toml`).
- Transcript mode switched from `legacy_messages` to `responses_events`.
- Client constants (`src/incident_handoff/client.py`) updated to match.
- Adapter (`src/incident_handoff/adapter.py`) now extracts text from Responses
  content blocks (array of `output_text` items) instead of a flat string.

## Event Ordering

Events are delivered in strict chronological order within a turn. The adapter
preserves the original sequence: `message` blocks appear before their
corresponding `tool_call`, which appears before the matching `tool_result`.
Replay serialization and deserialization maintain this order exactly.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`. The corresponding `tool_result`
event references the same `call_id`. The replay layer now preserves `call_id`
through serialize/deserialize round-trips so that tool results can be
correlated with their originating tool calls. This correlation is essential for
correct handoff and resume workflows.

## Replay

Replay remains event-sourced: `serialize_events` and `replay_from_serialized`
operate on the raw event stream. State is not rebuilt from rendered transcript
text.
