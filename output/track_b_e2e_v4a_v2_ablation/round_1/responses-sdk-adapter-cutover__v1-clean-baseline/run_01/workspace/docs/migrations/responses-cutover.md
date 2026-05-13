# Responses Cutover

## Overview

Migrate from the legacy chat-completions wrapper to the Responses wire path with event-sourced transcripts.

## Configuration

Update `config/runtime.toml`:

```toml
wire_api = "responses"
transcript_mode = "event"
```

## Event Ordering

- Events must be preserved in strict chronological order as emitted by the Responses API.
- Do not reorder events during serialization or replay.
- Each event carries an implicit sequence position determined by its position in the event stream.
- Tool calls and their corresponding results must maintain their original ordering relative to other events.

## Tool-Result Correlation

- Tool results are correlated to tool calls via the `call_id` field.
- The `call_id` is unique per tool invocation and must be preserved end-to-end.
- During replay, match `tool_result` events to their preceding `tool_call` events using `call_id`.
- Do not attempt to correlate tool results by name or arguments; rely solely on `call_id`.

## Replay Semantics

- Replay is event-sourced: reconstruct state by processing the ordered event stream.
- Do not rebuild state from rendered transcript text.
- Use `replay_from_serialized()` to deserialize events and restore state.
- Use `serialize_events()` to persist events for later replay.

## Migration Checklist

- [ ] Update `config/runtime.toml` to use `wire_api = "responses"`
- [ ] Set `transcript_mode = "event"` in config
- [ ] Verify event ordering is preserved in transcripts
- [ ] Confirm tool-result correlation via `call_id` works correctly
- [ ] Test replay from serialized events (not from rendered text)
