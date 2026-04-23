# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Read replay items from `response.output_item.added`.
- Preserve the stream-provided `sequence` and `call_id` on every tool call and tool result.
- Join tool results back to tool calls by `call_id`, not by tool name or visible ordinal.
- Keep repeated same-name tool invocations distinct in replay output.
- Keep the public CLI summary format stable as `tool_call[...]` / `tool_result[...]`.
- Treat `response.completed` as a footer marker after the per-item replay contract has been consumed.
