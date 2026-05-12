# Responses Cutover

## Overview

This workspace has migrated from the legacy chat-wrapper path to Responses event semantics.

## Key Changes

### Event Ordering

- Events are now consumed in strict chronological order as emitted by the Responses API.
- The sequence of events (e.g., `response.created`, `output_item.added`, `tool_call`, `tool_call_output`) must be preserved during replay.
- Do not reorder or batch events; each event must be applied in its original order to maintain correct state transitions.

### Tool-Result Correlation

- Tool calls and their results are correlated via unique call identifiers (`call_id` or equivalent).
- When replaying, ensure that each tool result is matched to its corresponding tool call using this identifier.
- Do not assume positional correlation; always use the explicit call ID to pair calls with results.

### Replay Behavior

- Replay is event-sourced: state is reconstructed by applying the sequence of events, not by parsing rendered transcript text.
- The transcript stores raw event objects, not rendered message text.
- This ensures deterministic replay and preserves all intermediate state changes.

## Migration Notes

- The legacy message wrapper has been removed.
- All code paths now expect Responses wire format and event-based transcripts.
- Existing transcript fixtures remain unchanged; only the interpretation logic has been updated.
