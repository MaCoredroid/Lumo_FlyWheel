# Responses Cutover

The runtime now follows the Responses-native replay contract.

- Read replay state from `response.output_item.added` items.
- Preserve the streamed `call_id` on every `tool_call` and `tool_result`.
- Join tool results back to calls by `call_id`, not `tool_name`.
- Keep repeated same-name tool calls as distinct replay steps when rendering.
- Continue treating `response.completed` as a stream footer, not as replay content.
- Keep the CLI summary format stable: assistant text first, then each tool call
  with its matching tool result.
