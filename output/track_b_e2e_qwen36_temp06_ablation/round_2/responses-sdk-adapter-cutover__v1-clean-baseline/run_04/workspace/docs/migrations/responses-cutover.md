# Responses Cutover

## Overview

Migrated from the legacy `chat_completions` chat-wrapper path to the Responses API wire path.

## Config Changes

- `config/runtime.toml` — `wire_api` set to `"responses"`, `transcript_mode` set to `"event_transcript"`.
- `.codex/config.toml` — updated to match `config/runtime.toml`.

## Source Changes

- `src/incident_handoff/client.py` — switched to Responses wire path; `extract_response_items` now reads `response["output"]` instead of the legacy `choices[0].message.content`.
- `src/incident_handoff/adapter.py` — `normalize_response_items` now extracts text from the `output_text` content array (`item["content"]` is a list of content parts) rather than reading a flat `item["content"]` string.
- `src/incident_handoff/replay.py` — `replay_from_serialized` now preserves `call_id` on both `tool_call` and `tool_result` events so tool-result correlation is maintained during replay.
- `src/incident_handoff/render.py` — `tool_result` rendering now includes the `call_id` in output for traceability.

## Event Ordering and Tool-Result Correlation

- **Event ordering is preserved**: the event-sourced transcript maintains the original sequence of `assistant_text`, `tool_call`, and `tool_result` events as emitted by the Responses API. Replay serializes and deserializes events in strict order — state is never rebuilt from rendered transcript text.
- **Tool-result correlation**: every `tool_call` event carries a `call_id`; the corresponding `tool_result` event references the same `call_id`. The replay layer now persists the `call_id` field during both serialization and deserialization so that tool calls and their results remain correlated after replay. This ensures that interleaved tool turns (multiple calls/results in a single turn) are reconstructed correctly.

## Transcript Fixtures

Transcript fixtures under `transcripts/` are unchanged. The adapter now correctly parses the `output_text` content parts used by the Responses API format.
