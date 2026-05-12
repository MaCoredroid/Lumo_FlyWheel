# Responses Cutover

## Overview

This workspace has migrated from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are processed in strict sequential order as they appear in the transcript.
- Each event carries a monotonically increasing sequence number.
- Replay must preserve the original event order; do not reorder or deduplicate events.
- Event-sourced replay reads must be performed in order to maintain causal consistency.

## Tool-Result Correlation

- Tool calls and their results are correlated via a stable `call_id` field.
- Each tool call event emits a unique `call_id`.
- The corresponding tool result event references the same `call_id`.
- Correlation must be preserved during replay; do not rely on `call_id` matching, not positional heuristics.

## Replay Behavior

- Replay is event-sourced: process raw events directly without rebuilding state from rendered transcript text.
- Do not parse or reconstruct state from message text; use the structured event payload.
- Maintain event fidelity to ensure deterministic replay.
