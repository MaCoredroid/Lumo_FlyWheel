# Responses Cutover

The runtime now follows the Responses-native replay contract.

- Read replay state from `response.output_item.added` items instead of a legacy assistant blob.
- Preserve the stream-provided `call_id` on every tool call and tool result.
- Join tool results back to tool calls by `call_id`, never by tool name or visible ordinal.
- Keep the public CLI summary lines stable as `tool_call[...]` and `tool_result[...]`.
- Treat `response.completed` as a footer marker only; it does not supply replay items.
