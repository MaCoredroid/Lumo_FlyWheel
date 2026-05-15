# Responses Cutover

## Overview

This document describes the migration from the legacy chat-wrapper path to Responses event semantics.

## Configuration Changes

- `wire_api` is now set to `responses` to use the Responses wire path
- `transcript_mode` is now set to `responses_events` for event-sourced replay

## Event Ordering

Events must be processed in strict chronological order as emitted by the Responses API. The event sequence preserves:

1. Message and tool invocation events maintain their original emission order
2. Each event carries a monotonic sequence number for deterministic replay
3. Reordering events during replay will break causal dependencies

## Tool-Result Correlation

Tool invocations and their results are correlated via the `call_id` field:

1. A tool call event contains a unique `call_id`
2. The corresponding tool result event references the same `call_id`
3. Replay must match results to calls using this identifier, not positional heuristics
4. Mismatched or missing `call_id` values indicate a transcript integrity issue

## Replay Semantics

Replay is event-sourced: state is reconstructed by processing the raw event stream, not by parsing rendered transcript text. This ensures:

- Fidelity to the original interaction
- Preservation of tool call/result pairing
- Deterministic, reproducible replays
