# Responses Cutover

The local adapter and reducer now follow the Responses-native replay contract.

- Read `response.output_item.added` items directly and preserve each tool call's
  emitted `call_id`.
- Join tool results back to tool calls by `call_id`, not by `tool_name`, so
  repeated same-name calls remain distinct through replay rendering.
- Preserve the public CLI summary line format as
  `tool_call[<call_id>] ...` / `tool_result[<call_id>] ...` for stable operator
  snapshots during the cutover.
- Treat `response.completed` as a footer marker only; replay state comes from
  the per-item Responses stream.
