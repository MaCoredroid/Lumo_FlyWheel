# Responses Cutover

This workspace now uses the Responses wire path and persists replayable event data
instead of the legacy chat-wrapper transcript shape.

## Required behavior

- Set `wire_api = "responses"` and `transcript_mode = "responses_events"`.
- Preserve event ordering exactly as emitted. Do not sort, regroup, or merge
  tool calls and tool results after normalization.
- Preserve tool-result correlation with `call_id` on both `tool_call` and
  `tool_result` events so rendered transcripts and replay can refer back to the
  same invocation.
- Keep replay event-sourced. Rebuild state from serialized event records, not
  from rendered transcript text.

## Event ordering

Responses output may interleave assistant text, tool calls, and tool results.
Normalization must emit internal events in the same order they appear on the
wire, including multiple assistant text blocks inside a single message item.

## Tool-result correlation

Every tool result must carry the originating `call_id`. Replay and rendering
must preserve that identifier so downstream consumers can match each result to
its initiating tool call without inferring from position or rendered text.
