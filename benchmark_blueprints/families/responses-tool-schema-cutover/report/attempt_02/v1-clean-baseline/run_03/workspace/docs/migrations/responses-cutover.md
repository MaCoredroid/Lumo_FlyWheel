# Responses Cutover

The replay path now follows the Responses output-item contract directly.

- Read replay data from `response.output_item.added` events.
- Preserve the stream-provided `call_id` on both `tool_call` and `tool_result` items.
- Join tool results back to calls by `call_id`, never by `tool_name` or synthetic ordinals.
- Keep the CLI replay summary stable by rendering calls in call order with their matching results attached, even when results arrive out of order in the stream.
- Treat `response.completed` as a terminal marker only; replay content comes from per-item output events.
