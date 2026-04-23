# Responses Cutover

The local adapter now follows the Responses replay contract.

- Read replay data from `response.output_item.added` events and inspect each embedded `item`.
- Preserve the stream-provided `item.call_id` for every `tool_call` and `tool_result`.
- Join tool results back to calls by `call_id`, not `tool_name`, so repeated same-name calls remain distinct.
- Keep the public CLI summary text stable: `tool_call[...]` and `tool_result[...]` lines still render in call order for successful runs.
- Treat `response.completed` as a footer marker after item-level replay has already been reconstructed.
