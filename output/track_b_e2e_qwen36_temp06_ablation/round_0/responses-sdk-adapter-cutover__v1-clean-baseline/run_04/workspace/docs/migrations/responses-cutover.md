# Responses Cutover

Migrated from the legacy chat-wrapper path to Responses event semantics.

## Changes

- `config/runtime.toml` — wire API set to `responses`, transcript mode set to `responses_events`.
- `src/incident_handoff/client.py` — wire config and response extraction updated to use Responses wire path (`response["output"]`).
- `src/incident_handoff/adapter.py` — normalized to handle Responses event format where message content is an array of parts (e.g., `output_text`).
- `src/incident_handoff/replay.py` — fixed `replay_from_serialized` to preserve `call_id` on tool_call and tool_result events.

## Event Ordering

Events are delivered in strict chronological order as emitted by the Responses API. The replay layer preserves this ordering verbatim — no reordering or deduplication is performed. Consumers must process events sequentially to maintain correct state.

## Tool-Result Correlation

Each `tool_call` event carries a `call_id`. The corresponding `tool_result` event references the same `call_id`. The replay serialization and deserialization round-trip preserves `call_id` on both event kinds so that correlation is maintained across serialization boundaries. Do not drop or reorder events between a `tool_call` and its matching `tool_result`.

## Replay

Replay remains event-sourced: state is reconstructed from the serialized event log, not from rendered transcript text. The `render_transcript` function is a read-only view and must not be used as the source of truth for replay.
