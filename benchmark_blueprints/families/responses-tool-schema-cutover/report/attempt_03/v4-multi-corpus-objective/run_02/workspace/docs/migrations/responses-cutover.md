# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Consume `response.output_item.added` items as the replay source of truth.
- Read tool arguments directly from each `tool_call` item.
- Join `tool_result` items back to `tool_call` items by the original `call_id`.
- Preserve repeated same-name tool invocations as distinct replay rows all the way through render output.
- Keep the public CLI summary text stable: assistant line first, then each tool call followed by its matched result in call order.
- Treat `response.completed` as a footer marker only; it does not replace per-item replay state.
