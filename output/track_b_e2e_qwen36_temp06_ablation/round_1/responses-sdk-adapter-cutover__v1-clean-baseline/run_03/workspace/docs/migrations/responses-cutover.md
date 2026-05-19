# Responses Cutover

This workspace has migrated from the legacy `chat_completions` wire path to the
Responses API (`responses`). The transcript mode is now `responses_events`.

## What Changed

- **Wire API**: `chat_completions` → `responses`
- **Transcript mode**: `legacy_messages` → `responses_events`
- **Client**: `extract_response_items` now reads `response["output"]` instead of
  walking `response["choices"][0]["message"]["content"]`.
- **Adapter**: `normalize_response_items` iterates over the `content` array inside each
  `message` item to extract `output_text` blocks.

## Event Ordering

Events are delivered in strict arrival order. The adapter preserves this order
when normalizing — `assistant_text`, `tool_call`, and `tool_result` events
appear in the same sequence as the Responses API emits them. Replay uses the
serialized event stream directly; it does not reorder or rebuild state from rendered
transcript text.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`. The corresponding `tool_result`
event references the same `call_id`. This correlation is preserved end-to-end:

- The adapter passes `call_id` through unchanged.
- Serialization stores `call_id` as the second pipe-delimited field for both
  `tool_call` and `tool_result` lines.
- Deserialization restores `call_id` so that replay roundtrips are lossless.

Rendered transcripts include `call_id` in both tool-call and tool-result lines
(e.g. `tool_result(call-1): oncall-primary`) so that humans can trace
correlation without inspecting raw events.

## Replay

Replay remains event-sourced. `replay_from_serialized` reconstructs the event
list from the serialized pipe-delimited format; it never parses rendered
transcript text to rebuild state.
