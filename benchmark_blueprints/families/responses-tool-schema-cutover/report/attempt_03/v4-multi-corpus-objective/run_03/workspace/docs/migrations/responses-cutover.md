# Responses Cutover

The local replay path now follows the Responses-native output-item contract.

- Read assistant text from `response.output_item.added` message items whose content parts are `output_text`.
- Preserve every `tool_call` and `tool_result` item as its own replay row, even when repeated calls share the same `tool_name`.
- Join tool results back to tool calls by the stream-provided `call_id`. Do not dedupe by tool name and do not synthesize replacement ids from ordinals or fixture position.
- Keep the public CLI summary format stable for successful runs: assistant text first, then each tool call followed by its matching tool result in call order.
- Treat `response.completed` as the stream terminator only; replay state is derived from the per-item Responses events.
