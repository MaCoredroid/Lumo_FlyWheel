# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Read replay fixtures from `response.output_item.added` events.
- Preserve each `tool_call` and `tool_result` as a distinct item keyed by the stream-provided `call_id`.
- Join tool results back to calls by `call_id`, not by `tool_name`, so repeated same-name calls stay distinct.
- Keep the public CLI summary stable by rendering assistant text first, then each tool call with its matched result in original call order.
- Treat `response.completed` as a terminal marker only; the replay contract lives in the per-item stream.
