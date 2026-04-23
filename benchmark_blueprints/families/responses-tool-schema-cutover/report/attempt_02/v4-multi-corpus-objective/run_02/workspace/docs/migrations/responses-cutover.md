# Responses Cutover

The local adapter and reducer now follow the native Responses replay contract.

- Read replay state from `response.output_item.added` items rather than a legacy
  assistant blob.
- Preserve each tool invocation's stream-provided `call_id`; repeated same-name
  tool calls remain distinct in normalized output, replay joins, and rendered
  summaries.
- Join tool results back to tool calls by `call_id`, never by tool name or a
  synthetic ordinal.
- Keep the public CLI summary line format stable as
  `tool_call[...]` / `tool_result[...]` during the cutover.
- Treat `response.completed` as a footer marker only; replay content comes from
  the per-item stream contract.
