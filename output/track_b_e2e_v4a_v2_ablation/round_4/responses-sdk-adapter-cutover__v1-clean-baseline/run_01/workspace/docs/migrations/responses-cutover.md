# Responses Cutover

This document covers the migration from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

Events in the Responses wire path are delivered in strict chronological order. Each event carries a monotonically increasing sequence number that must be preserved during replay. Do not reorder, deduplicate, or batch events—process them exactly as received to maintain causal consistency.

## Tool-Result Correlation

Tool calls and their corresponding results are correlated via the `call_id` field. When replaying transcripts:

- Each tool call event includes a unique `call_id`.
- The matching tool result event references the same `call_id`.
- Preserve this correlation during replay; do not attempt to re-match based on rendered text or positional heuristics.

## Replay Mode

Replay remains event-sourced. Do not rebuild state from rendered transcript text. Instead, re-emit the original event stream in order, allowing the system to reconstruct state deterministically from the event log.
