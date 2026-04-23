# Responses Cutover

The replay runtime now follows the Responses-native output-item contract.

- Read replay state from `response.output_item.added` items.
- Preserve the stream-provided `call_id` on both `tool_call` and `tool_result`.
- Join tool results back to calls by `call_id`, not by tool name or ordinal position.
- Keep repeated same-name tool calls distinct all the way through rendered replay output.
- Treat `response.completed` as a terminal marker only; it does not replace per-item replay state.
