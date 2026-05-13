# Responses Cutover

## Overview

This workspace has migrated from the legacy chat-wrapper path to Responses event semantics.

## Changes

- Wire API

- `wire_api` is now set to `responses` instead of `chat_completions`.
- `transcript_mode` is now set to `responses_events` instead of `legacy_messages`.

## Event Ordering

Events are now processed in strict chronological order as they appear in the Responses event stream. The order of events in the transcript is preserved exactly:

1. `assistant_text` events represent assistant message content.
2. `tool_call` events represent function invocation requests with `call_id`, `name`, and `arguments`.
3. `tool_result` events represent function execution results with `call_id` and `output`.

The sequence of events must not be reordered during serialization, deserialization, or replay.

## Tool-Result Correlation

Tool calls and their results are correlated via the `call_id` field:

- Each `tool_call` event includes a unique `call_id`.
- The corresponding `tool_result` event references the same `call_id`.
- This correlation is preserved through serialization and replay to ensure accurate reconstruction of the conversation flow.

## Replay Semantics

Replay is event-sourced: events are deserialized from the transcript format directly without rebuilding state from rendered text. This ensures:

- Exact event ordering is maintained.
- Tool-result correlation via `call_id` is preserved.
- No information is lost during replay.
