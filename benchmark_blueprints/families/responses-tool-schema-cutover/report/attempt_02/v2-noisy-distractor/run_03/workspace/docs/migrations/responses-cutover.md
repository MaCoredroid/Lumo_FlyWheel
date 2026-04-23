# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Read streamed `response.output_item.added` items as the replay source of truth.
- Preserve the original `call_id` on every `tool_call` and `tool_result`.
- Join tool results back to calls by `call_id`, not by tool name or stream position.
- Keep the CLI summary shape stable: assistant text first, then each tool call with its matched tool result in call order.
- Treat `response.completed` as a terminal marker only; it does not replace the per-item replay contract.
