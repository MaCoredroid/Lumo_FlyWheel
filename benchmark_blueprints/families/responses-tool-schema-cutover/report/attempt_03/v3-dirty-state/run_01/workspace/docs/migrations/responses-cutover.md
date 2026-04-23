# Responses Cutover

The local adapter now follows the Responses replay contract.

- Read replayable items from `response.output_item.added`.
- Preserve every tool invocation by the stream-provided `call_id`.
- Join `tool_result` items back to `tool_call` items by `call_id`, even when repeated calls share the same `tool_name` and results arrive out of order.
- Keep the public CLI summary format stable: `assistant: ...`, `tool_call[call_id] ...`, and `tool_result[call_id] ...`.
- Treat `response.completed` as a terminal footer marker, not as the source of replay state.
