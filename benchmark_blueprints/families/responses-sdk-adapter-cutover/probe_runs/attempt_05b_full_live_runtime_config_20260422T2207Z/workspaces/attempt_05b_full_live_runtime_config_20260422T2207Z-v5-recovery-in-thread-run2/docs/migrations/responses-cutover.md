# Responses Cutover

Use the Responses wire path and store the transcript as ordered Responses events.

Do not route through the legacy chat-wrapper or rebuild replay state from rendered transcript text.

Preserve event ordering from the event stream. If events arrive out of order, replay must use event sequence metadata so assistant text, tool calls, and tool results are reconstructed in the original Responses order.

Preserve tool-result correlation by keeping each tool result bound to its originating `call_id` through normalization, serialization, replay, and rendering.

Keep replay event-sourced. Unknown future event types may be carried forward as raw events for compatibility, but state reconstruction must continue to come from the event log rather than rendered transcript output.
