# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Replay `response.output_item.added` items in stream order and preserve their native
  `sequence` values when normalizing.
- Read tool arguments from `item.arguments` on `tool_call` items.
- Join each `tool_result` back to its originating `tool_call` by the native `call_id`.
- Keep repeated same-name tool invocations distinct all the way through replay render
  output; never dedupe by tool name or visible ordinal.
- Keep the public CLI summary format stable as `tool_call[...]` / `tool_result[...]`
  lines during the Responses cutover.
- Treat `response.completed` as a footer marker only; replay state comes from per-item
  `response.output_item.added` events.
