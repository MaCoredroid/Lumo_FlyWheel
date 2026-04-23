# Responses Cutover

The runtime now follows the native Responses replay contract.

- Read `response.output_item.added` events and preserve each tool call/result as its own replay item.
- Recover tool arguments from `item.arguments` and join tool results back to tool calls by the original `call_id`.
- Keep repeated same-name tool invocations distinct all the way through render output; never dedupe by `tool_name` or synthesize replacement ids.
- Treat `response.completed` as a footer marker only; it does not replace the per-item replay stream.
- Keep the successful CLI summary lines stable: assistant text, then `tool_call[...]`, then `tool_result[...]`.
