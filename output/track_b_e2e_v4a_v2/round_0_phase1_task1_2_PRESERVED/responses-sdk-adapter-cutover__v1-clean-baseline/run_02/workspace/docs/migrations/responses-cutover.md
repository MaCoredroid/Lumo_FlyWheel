# Responses Cutover

## Overview

This document covers the migration from the legacy chat-wrapper path to Responses event semantics.

## Event Ordering

- Events are processed in strict sequential order as they appear in the transcript.
- Each event carries a monotonically increasing sequence identifier.
- Replay must preserve the original event order to maintain causal consistency.
- Do not reorder or batch events during replay; process them in the order recorded.

## Tool-Result Correlation

- Tool calls and their results are correlated via a unique `call_id` field.
- Each tool_call event includes a `call_id` that matches its corresponding tool_result event.
- During replay, maintain a mapping of pending tool calls by `call_id` to correlate results.
- Tool results must be applied only after their corresponding tool_call event has been processed.

## Replay Semantics

- Replay is event-sourced: state is rebuilt by processing the event stream, not from rendered transcript text.
- Do not attempt to reconstruct state by parsing rendered message text.
- The event stream is the single source of truth for replay operations.

## Configuration

- The required config file is `config/runtime.toml`.
- Set `wire_api = "responses"` to use the Responses wire path.
- Set `transcript_mode = "responses_events"` to enable event-sourced replay.
