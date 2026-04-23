# Responses Cutover

The replay runtime now follows the Responses-native stream contract.

- Read replay state from `response.output_item.added` items rather than legacy Chat Completions blobs.
- Preserve each `tool_call` and `tool_result` as a distinct output item and keep the stream-provided `call_id` unchanged.
- Join tool results back onto calls by `call_id`, not by `tool_name`, because the same tool may be invoked multiple times in one response.
- Keep the public CLI summary lines stable: assistant text first, then each `tool_call[...]` immediately followed by its matching `tool_result[...]` when a result exists.
- Treat `response.completed` as terminal metadata only; it does not replace the per-item replay contract.
