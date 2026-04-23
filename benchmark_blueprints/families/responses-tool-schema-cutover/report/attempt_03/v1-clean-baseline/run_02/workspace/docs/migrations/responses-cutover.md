# Responses Cutover

The local adapter now follows the Responses output-item replay contract.

- Read replay state from `response.output_item.added` events in sequence order.
- Preserve every tool call and tool result as its own replay item, keyed by the original `call_id`.
- Join tool results back to tool calls by `call_id`, not by `tool_name` or fixture position.
- Keep the public CLI summary format unchanged for successful runs: assistant text followed by each tool call and its matched tool result.
- Treat `response.completed` as the response footer only; replay content comes from the per-item stream.
