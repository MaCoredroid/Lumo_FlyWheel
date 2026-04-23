# Responses Cutover

This workspace now runs on the Responses wire path and stores transcripts as
Responses events, not legacy chat-wrapper messages.

## Required runtime settings

- `wire_api = "responses"`
- `transcript_mode = "responses_events"`

## Ordering and correlation rules

- Preserve the original event stream order exactly as emitted.
- Keep tool calls and tool results correlated by `call_id`.
- Rendered transcript text is for display only; replay must consume serialized
  events rather than reconstructing state from rendered output.
- Tool results must remain attached to the matching tool call even when
  assistant text, tool calls, and tool results are interleaved in the same
  turn.

## Replay contract

Replay remains event-sourced:

- serialize the normalized event objects directly
- replay by reading those serialized event objects back in order
- do not rebuild tool state from formatted transcript lines
