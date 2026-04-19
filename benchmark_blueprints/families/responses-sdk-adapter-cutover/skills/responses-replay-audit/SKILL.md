# Responses Replay Audit

Use this skill when the task involves Responses-style event streams, transcript replay, or tool-result ordering.

## Workflow
1. Inspect raw event objects before touching the renderer.
2. Confirm replay derives from serialized events, not rendered transcript text.
3. Preserve tool-call IDs and tool-result correlation through render and replay.
4. Treat config and docs as downstream of the event model, not as the primary fix.

## Anti-Patterns
- Flattening assistant output into final text.
- Routing tools from rendered transcript strings.
- Updating fixtures without fixing event semantics.

## Done Signal
- Visible tests pass and replay remains event-sourced.
