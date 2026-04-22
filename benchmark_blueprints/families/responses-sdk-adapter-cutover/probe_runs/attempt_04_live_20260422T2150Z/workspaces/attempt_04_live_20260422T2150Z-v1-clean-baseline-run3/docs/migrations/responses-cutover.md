# Responses Cutover

This workspace now uses the Responses wire path and persists transcript data as
Responses events.

## Required config

- Set `wire_api = "responses"`.
- Set `transcript_mode = "responses_events"`.

## Event ordering

Replay must consume the stored event stream in the exact order emitted by the
Responses API. Do not collapse the stream into a rendered transcript and do not
re-sort tool activity after normalization.

## Tool-result correlation

Tool results must remain correlated to the originating tool call via `call_id`.
Preserve `call_id` on both `tool_call` and `tool_result` events through
normalization, serialization, replay, and rendering so interleaved tool traffic
stays attributable to the correct invocation.

## Event-sourced replay

Replay stays event-sourced. Serialize and hydrate the normalized event objects
directly, and rebuild state from those stored events rather than from rendered
transcript text.
