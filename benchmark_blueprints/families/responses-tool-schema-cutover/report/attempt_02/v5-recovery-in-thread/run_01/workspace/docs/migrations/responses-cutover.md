# Responses Cutover

The replay path now follows the native Responses stream contract.

- Read tool calls and tool results from `response.output_item.added` events.
- Preserve the stream-provided `item.call_id` for every tool invocation and join
  results back to calls by that exact `call_id`.
- Do not collapse repeated same-name tool invocations into one logical step.
- Keep the public CLI summary format stable as `tool_call[...]` /
  `tool_result[...]`.
- Treat `response.completed` as a footer marker only; replay state comes from the
  per-item output stream.
