# Responses Cutover

The local replay path now follows the Responses-native output-item contract.

- Read replay data from `response.output_item.added` events and preserve each item's stream `sequence`.
- Recover tool arguments from `item.arguments` on `tool_call` items.
- Join `tool_result` items back to their originating `tool_call` by the stream-provided `call_id`.
- Do not collapse repeated same-name tool invocations; multiple calls to the same tool remain distinct when their `call_id` values differ.
- Preserve the public CLI summary shape for successful runs:
  `assistant: ...`, `tool_call[<call_id>] ...`, and `tool_result[<call_id>] ...`.
- Treat `response.completed` as a terminal footer only; it does not replace per-item replay state.
