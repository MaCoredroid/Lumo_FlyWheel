# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Changes

- Wire API: `chat_completions` → `responses`
- Transcript mode: `legacy_messages` → `responses_events`

## Event Ordering

Events are processed in strict arrival order. The event stream maintains causal ordering:

1. `response.created` - Initial response event
2. `response.output_item.added` - Each new item (message, tool call)
3. `response.output_item.done` - Final response completion

Replay must preserve this order; do not reorder or deduplicate events.

## Tool-Result Correlation

Tool calls and their results are correlated via `call_id`:

- `response.function_call` events contain a `call_id`
- `response.function_call_output` events reference the same `call_id`

When replaying:
- Match tool results to calls using `call_id`
- Preserve the temporal sequence of call → result
- Do not attempt to rebuild state from rendered transcript text; use raw events

## Replay

Replay is event-sourced. Each event in the transcript represents a state transition. Process events sequentially to reconstruct conversation state.
