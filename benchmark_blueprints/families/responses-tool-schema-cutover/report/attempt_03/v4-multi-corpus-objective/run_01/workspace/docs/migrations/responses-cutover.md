# Responses Cutover

The local adapter now follows the Responses-native replay contract.

- Read replay state from `response.output_item.added` items instead of reconstructing a legacy assistant blob.
- Preserve the stream-provided `call_id` on every `tool_call` and `tool_result`; do not synthesize ids from ordinals or fixture position.
- Join `tool_result` rows back to `tool_call` rows by `call_id`, not by `tool_name`, so repeated same-name invocations remain distinct.
- Keep the public CLI summary format stable as `tool_call[...]` / `tool_result[...]` while pairing results in call order for diff-friendly operator snapshots.
- Treat `response.completed` as a completion footer only; replay content comes from the per-item stream.
