# Responses Cutover

The local replay path now follows the Responses-native output-item contract.

- Read `response.output_item.added` events and normalize each emitted item by its
  stream payload, not by Chat Completions-era fields.
- Preserve the original `call_id` on every `tool_call` and `tool_result`.
- Join tool results back to calls by `call_id`, even when repeated calls share the
  same `tool_name` and results arrive out of order.
- Keep the public CLI summary format stable by rendering calls in call order with
  the matching result placed directly after each call.
- Treat `response.completed` as a response footer only; replay state comes from
  the per-item stream events.
