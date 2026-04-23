# Responses Cutover

Use the Responses wire path and persist the raw Responses event stream as the
source of truth for replay.

## Required runtime settings

- Set `wire_api = "responses"` in `config/runtime.toml`.
- Set `transcript_mode = "responses_events"` in `config/runtime.toml`.

## Event semantics

- Normalize `response.output` items directly instead of reading a legacy
  `choices[].message` wrapper.
- Preserve item order exactly as emitted so assistant text, tool calls, and
  tool results remain interleaved in the original sequence.
- Keep tool-result correlation keyed by `call_id`; do not infer matches from
  rendered transcript text or adjacency alone.

## Replay requirements

- Replay must stay event-sourced from the serialized event log.
- Store and restore complete event records so `call_id`, tool name,
  arguments, and tool output survive a round trip intact.
- Rendering is for display only; do not rebuild replay state from rendered
  transcript lines.
