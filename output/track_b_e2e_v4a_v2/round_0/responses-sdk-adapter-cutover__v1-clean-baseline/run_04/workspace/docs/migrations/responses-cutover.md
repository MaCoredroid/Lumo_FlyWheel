# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Changes

- Wire API: `chat_completions` → `responses`
- Transcript mode: `legacy_messages` → `responses_events`

## Event Ordering

Events must be processed in strict sequential order as emitted order. Each event carries a monotonically increasing sequence number that defines its position in the stream. Do not reorder, drop, or batch events during replay; preserve the original emission order to maintain causal consistency.

## Tool-Result Correlation

Tool calls and their results are correlated via a stable `call_id` field:

- Tool call events emit a `call_id` when invoking a tool.
- The corresponding tool result event references the same `call_id`.
- During replay, match results to calls by this ID to ensure correct correlation.
- Do not infer tool outcomes from rendered transcript text; rely solely on the event stream.

## Replay

Replay remains event-sourced: reconstruct state by reapplying the event stream, not by parsing rendered transcript text.
