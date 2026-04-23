# Responses Cutover

The runtime now follows the Responses-native replay contract.

- Read replay items from `response.output_item.added` events.
- Consume `item.type == "message"` for assistant text and `item.type == "tool_call"` / `item.type == "tool_result"` for tool traffic.
- Join tool results back to tool calls by the stream-provided `call_id`, not by `tool_name` and not by synthetic ordinals.
- Keep repeated same-name tool calls distinct all the way through replay rendering.
- Preserve the successful CLI summary line format while emitting one call/result pair per original tool invocation.
- Treat `response.completed` as a footer marker only; replay state comes from the per-item stream contract above.
