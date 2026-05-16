# Responses Cutover

Migrate from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events must be processed in strict sequence order as emitted by the Responses API.
- Each event carries a monotonically increasing sequence number for replay.
- Do not reorder, drop, or deduplicate events during transcript replay.
- Preserve the original event stream order to maintain causal relationships.

## Tool-Result Correlation

- Tool calls and their results are correlated via `call_id` (or equivalent identifier).
- Each tool_result event must reference the original tool_call event it completes.
- Maintain the correlation ID through transcript serialization and replay.
- Do not decouple tool results from their originating calls during migration.

## Replay Semantics

- Keep replay event-sourced; process raw events rather than rendered output.
- Do not rebuild state from rendered transcript text.
- Use the event stream as the single source of truth for state reconstruction.
