# Responses Cutover

Use the Responses wire path and persist the Responses event stream directly.

- Set `wire_api = "responses"` and `transcript_mode = "responses_events"` in `config/runtime.toml`.
- Read response items from the Responses `output` array instead of the legacy chat-wrapper `choices[0].message.content` path.
- Preserve event ordering exactly as emitted. Replay must consume the serialized event stream in-order; do not rebuild state from rendered transcript text.
- Preserve tool-result correlation by carrying `call_id` through adapter normalization, replay serialization, replay deserialization, and transcript rendering.
- Treat rendered transcript text as a view over the event log only. It is not the replay source of truth.
