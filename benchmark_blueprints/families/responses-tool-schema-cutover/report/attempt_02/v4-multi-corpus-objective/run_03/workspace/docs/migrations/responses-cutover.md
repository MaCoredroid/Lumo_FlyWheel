# Responses Cutover

The local replay path now follows the Responses output-item contract.

- Consume `response.output_item.added` events as the replay source of truth.
- Read tool arguments from each `tool_call` item and join `tool_result` items by the original stream `call_id`.
- Preserve repeated same-name tool invocations as distinct replay rows; do not collapse them by `tool_name`.
- `response.completed` remains a footer marker. It does not replace the per-item replay contract.
- Keep the operator-facing CLI summary format stable as `tool_call[...]` / `tool_result[...]` while the underlying transport is Responses-native.
