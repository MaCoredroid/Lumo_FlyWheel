# Responses Cutover

The replay runtime now follows the Responses-native per-item stream contract.

- Consume `response.output_item.added` events and read tool payloads from `item`.
- Preserve every `tool_call` and `tool_result` entry by its original `call_id`.
- Join tool results back to tool calls with `call_id`, not `tool_name`.
- Keep repeated same-name tool invocations distinct all the way through replay render output.
- Keep the public CLI summary format unchanged; only the join semantics move to Responses-native routing.
