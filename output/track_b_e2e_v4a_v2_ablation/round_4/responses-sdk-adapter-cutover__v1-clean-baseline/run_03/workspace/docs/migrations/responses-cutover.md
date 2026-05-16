# Responses Cutover

## Overview

This migration transitions the system from the legacy chat-completions wrapper to the Responses API with event-sourced transcript semantics.

## Configuration Changes

- `wire_api`: Changed from `chat_completions` to `responses`
- `transcript_mode`: Changed from `legacy_messages` to `events`

## Event Ordering

Events must be processed in strict sequential order as emitted by the Responses API. Each event carries a monotonically increasing sequence identifier that must be preserved during:

- Initial streaming consumption
- Transcript persistence to storage
- Replay operations

Do not reorder, deduplicate, or batch events. The event sequence is the source of truth for conversation state reconstruction.

## Tool-Result Correlation

Tool calls and their results are correlated via the `call_id` field:

1. A `tool_call` event emits a unique `call_id`
2. The corresponding `tool_call_result` event references the same `call_id`
3. During replay, maintain the correlation to ensure results are applied to the correct invocation

Do not attempt to correlate tool calls and results by parsing rendered transcript text. Always use the structured event fields.

## Replay Semantics

Replay is event-sourced: reconstruct conversation state by replaying the raw event stream, not by parsing rendered transcript text. This ensures:

- Deterministic state reconstruction
- Preservation of intermediate states (e.g., partial tool calls)
- Correct handling of cancellations and retried invocations
