# Responses Cutover

The gateway now replays Responses-native event items end to end.

- Read each `response.output_item.added` record as its own replay input.
- Preserve every repeated tool invocation as a distinct call keyed by `call_id`.
- Join tool results back to tool calls by `call_id`, not by tool name.
- `response.completed` closes the stream but does not replace the per-item replay contract.
