# Responses Cutover

Cut the adapter over to the Responses wire and persist transcripts as ordered
Responses events.

Set both workspace runtime configs to:

- `wire_api = "responses"`
- `transcript_mode = "responses_events"`

Replay must remain event-sourced. Rebuild state from the serialized event stream
itself, in order, and do not reconstruct state by parsing rendered transcript
text.

Preserve event ordering exactly as emitted. Assistant output blocks, tool calls,
and tool results must remain in stream order so replay and rendering observe the
same sequence produced on the wire.

Preserve tool-result correlation via `call_id` on both `tool_call` and
`tool_result` events. Replay and render paths must carry that identifier
through, which keeps every tool result matched to its originating tool call
without consulting rendered transcript text.
