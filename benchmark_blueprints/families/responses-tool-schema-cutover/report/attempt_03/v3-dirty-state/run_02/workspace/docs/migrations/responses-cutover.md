# Responses Cutover

The replay adapter now follows the Responses-native stream contract.

- Read replay data from `response.output_item.added` events.
- Recover assistant text from `item.type == "message"` and `content[].type == "output_text"`.
- Recover tool calls from `item.type == "tool_call"` and tool results from `item.type == "tool_result"`.
- Join tool results back to tool calls by the stream-provided `item.call_id`.
- Preserve repeated same-name tool calls as distinct replay rows; never collapse or renumber them by tool name or ordinal position.
- Keep the CLI summary format unchanged for successful runs: assistant line first, then one `tool_call[...]` / `tool_result[...]` pair per original tool invocation.
- Treat `response.completed` as a footer marker only; it does not replace the per-item replay stream.
