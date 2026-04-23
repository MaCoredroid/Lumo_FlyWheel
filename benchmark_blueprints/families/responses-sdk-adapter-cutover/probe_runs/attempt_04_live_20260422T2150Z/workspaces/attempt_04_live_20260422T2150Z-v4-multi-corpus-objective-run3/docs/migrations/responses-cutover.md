# Responses Cutover

Use the Responses wire path and persist transcripts in `responses_events` mode.

Treat the transcript as an ordered event log, not as rendered chat text. Replay must
round-trip serialized events directly so assistant text, tool calls, and tool
results keep their original ordering without reconstructing state from a rendered
transcript.

Preserve tool-result correlation with the originating `call_id` on both tool call
and tool result events. When Responses items arrive with explicit sequence numbers,
normalize in sequence order before rendering or replay so interleaved assistant and
tool activity stays stable across handoff and audit flows.
