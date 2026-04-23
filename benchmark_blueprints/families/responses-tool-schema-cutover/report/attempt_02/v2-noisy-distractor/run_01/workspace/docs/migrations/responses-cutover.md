# Responses Cutover

The local replay path now follows the Responses-native streaming contract.

- Read replay state from `response.output_item.added` events and their `item` payloads.
- Recover tool calls from `item.type == "tool_call"` and tool results from `item.type == "tool_result"`.
- Preserve the original `call_id` from the Responses stream and use it as the only join key between a tool call and its result.
- Do not dedupe repeated tool invocations by `tool_name`; same-name calls remain distinct replay rows when their `call_id` values differ.
- `response.completed` remains a footer marker. It does not replace the per-item replay contract.
- Keep the successful CLI summary format stable:
  `assistant: ...`
  `tool_call[<call_id>] <tool_name> <arguments>`
  `tool_result[<call_id>] <tool_name> => <output>`
