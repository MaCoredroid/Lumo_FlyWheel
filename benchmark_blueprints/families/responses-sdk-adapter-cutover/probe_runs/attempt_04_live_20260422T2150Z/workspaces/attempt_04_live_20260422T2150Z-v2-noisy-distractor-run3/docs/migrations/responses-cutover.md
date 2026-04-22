# Responses Cutover

This workspace now reads the Responses wire path directly and records
`responses_events` transcripts instead of relying on the legacy chat wrapper.

Migration requirements:
- Read response items from the Responses `output` stream in arrival order.
- Normalize message content from event blocks, preserving the original event
  sequence instead of collapsing the turn into rendered transcript text.
- Preserve `call_id` on both `tool_call` and `tool_result` events so tool
  results stay correlated with the exact invocation that produced them.
- Keep replay event-sourced: serialize and replay the structured events
  themselves, not a rendered transcript view reconstructed back into state.
- Treat transcript rendering as a presentation layer only; it must not become
  the source of truth for replay or tool-result matching.
