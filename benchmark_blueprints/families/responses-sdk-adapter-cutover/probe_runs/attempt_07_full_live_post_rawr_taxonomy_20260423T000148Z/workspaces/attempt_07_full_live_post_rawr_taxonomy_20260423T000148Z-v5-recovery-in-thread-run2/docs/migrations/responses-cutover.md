# Responses Cutover

Use the Responses wire path and store transcripts as ordered response events.

Replay must stay event-sourced: deserialize the recorded event stream directly and do not rebuild state from rendered transcript text.

Preserve event ordering exactly as emitted. Do not regroup tool activity under rendered assistant turns or reorder tool results after rendering.

Correlate each tool result with its originating tool call through the shared `call_id`. Rendering may summarize events for humans, but replay and audit logic must retain the original event sequence and `call_id` linkage.
