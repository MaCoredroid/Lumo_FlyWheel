# Responses Cutover

The replay path now follows the Responses-native stream contract.

- Read tool activity from `response.output_item.added` events.
- Recover assistant text from `item.type == "message"` content parts with `type == "output_text"`.
- Recover tool invocations from `item.type == "tool_call"` and preserve the original `call_id`.
- Recover tool outputs from `item.type == "tool_result"` and join them back to calls by `call_id`.
- Do not collapse repeated same-name tool calls; distinct `call_id` values remain distinct replay rows end to end.
- Keep the public CLI summary stable by rendering tool calls in call order and attaching the matching result for each successful call.
- Treat `response.completed` as a footer marker only; replay state comes from per-item output events.
