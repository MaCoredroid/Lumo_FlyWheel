# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Rebuild replays from `response.output_item.added` records and inspect `item.type`
  to recover `message`, `tool_call`, and `tool_result` items.
- Join tool results back to calls by the original `call_id` from the stream.
  Repeated same-name tool invocations remain distinct and must not be deduped by
  `tool_name` or by visible ordinal position.
- Keep the public CLI summary output stable: render assistant text first, then
  emit each tool call in call order with its matched tool result when present.
- Treat `response.completed` as a footer marker only after the per-item replay
  state has already been reconstructed from output items.
