# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

Events are replayed in their original sequence order. Each event carries a monotonically increasing sequence number that must be preserved during replay. Do not reorder events or rebuild state from rendered transcript text; replay must remain event-sourced.

## Tool-Result Correlation

Tool calls and their results are correlated via a stable `call_id`. When replaying:
- Match each `tool_call` event with its corresponding `tool_result` event using `call_id`.
- Preserve the causal ordering: tool results must follow their invocations.
- Do not infer or reconstruct tool results from transcript renderings.
