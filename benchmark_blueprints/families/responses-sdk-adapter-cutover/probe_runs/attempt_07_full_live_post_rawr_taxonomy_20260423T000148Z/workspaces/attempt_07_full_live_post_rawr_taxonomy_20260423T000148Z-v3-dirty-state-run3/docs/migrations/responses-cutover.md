# Responses Cutover

Use the Responses wire path and store the transcript in Responses event mode.

Replay stays event-sourced. Serialize and replay the normalized event stream directly; do not rebuild state from rendered transcript text.

Ordering rules:
- Normalize items in event order. If a `sequence` field is present, use it to restore the original Responses item order before emitting events.
- Preserve message content block order when flattening `output_text` blocks into assistant text events.

Tool correlation rules:
- Carry `call_id` through both `tool_call` and `tool_result` events.
- Keep rendered transcripts and serialized replay records annotated with the same `call_id` so tool results stay correlated to the originating tool call.
