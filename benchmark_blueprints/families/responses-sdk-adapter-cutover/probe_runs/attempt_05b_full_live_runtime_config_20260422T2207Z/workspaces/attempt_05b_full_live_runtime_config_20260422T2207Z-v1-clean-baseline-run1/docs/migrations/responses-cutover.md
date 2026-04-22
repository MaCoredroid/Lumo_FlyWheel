# Responses Cutover

This workspace is cut over to the Responses wire path and Responses event
transcript mode.

## Required runtime settings

- `wire_api = "responses"`
- `transcript_mode = "responses_events"`

## Event normalization

- Normalize the raw Responses output stream into ordered event records.
- Emit assistant text from message content blocks in the same order they appear.
- Preserve tool calls and tool results as distinct events with their original
  `call_id`.

## Ordering and correlation

- Replay must consume the serialized event stream directly and preserve event
  ordering exactly as recorded.
- Tool results must stay correlated to the originating tool call through the
  shared `call_id`.
- Do not infer tool-result linkage from rendered transcript text or positional
  guesses.

## Replay contract

- Keep replay event-sourced.
- Serialize and deserialize the event records themselves.
- Do not rebuild runtime state from a human-readable transcript render.
