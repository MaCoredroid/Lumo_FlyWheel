# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Replay from `response.output_item.added` items in stream `sequence` order.
- Preserve the Responses `call_id` exactly as emitted; do not synthesize ordinals.
- Join `tool_call` and `tool_result` items by `call_id`, not by `tool_name`.
- Keep repeated same-name tool invocations distinct all the way through replay render output.
- `response.completed` remains a footer marker and does not replace per-item replay state.
- Keep the public CLI summary lines stable as `tool_call[...]` / `tool_result[...]` for successful runs.
