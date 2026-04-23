# Responses Cutover

Use the Responses wire path and persist the transcript as ordered response events.

Keep replay event-sourced: serialize and replay the normalized event stream itself,
not rendered transcript text.

Preserve event ordering exactly as emitted by Responses. Do not regroup message
blocks around tool activity, and do not reorder tool calls or tool results during
normalization, serialization, replay, or rendering.

Preserve tool-result correlation by carrying each `tool_result.call_id` end to end.
Replay must restore the same `call_id` so every tool result remains attached to the
tool call that produced it.
