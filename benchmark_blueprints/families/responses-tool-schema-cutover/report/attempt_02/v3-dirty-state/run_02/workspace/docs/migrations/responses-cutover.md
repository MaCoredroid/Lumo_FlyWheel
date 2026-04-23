# Responses Cutover

The replay path now follows the Responses-native output-item contract.

- Read replay state from `response.output_item.added` events and preserve each item's stream `sequence`.
- Recover tool arguments from `item.arguments` on `tool_call` items.
- Join `tool_result` items back to calls by the original `call_id`, not by tool name or stream ordinal.
- Preserve repeated same-name tool calls as distinct transcript rows all the way through render output.
- Treat `response.completed` as a footer marker only; replay state comes from the emitted output items.
