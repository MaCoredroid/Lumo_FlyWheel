# Responses Cutover

The runtime now follows the Responses-native replay contract.

- Read `response.output_item.added` items directly from the stream and preserve their
  stream `sequence` for normalization.
- Recover tool calls from `item.type == "tool_call"` and tool results from
  `item.type == "tool_result"`.
- Join tool results to tool calls by the original stream `call_id`, not by
  `tool_name` and not by synthetic ordinals.
- Keep repeated same-name tool calls distinct through replay rendering and CLI
  summaries by carrying the original `call_id` end to end.
- Keep the public CLI summary format stable as `tool_call[...]` and
  `tool_result[...]` lines while the underlying transport uses Responses-native
  output items.
- Treat `response.completed` as a terminal footer only; it does not replace the
  per-item replay contract.
