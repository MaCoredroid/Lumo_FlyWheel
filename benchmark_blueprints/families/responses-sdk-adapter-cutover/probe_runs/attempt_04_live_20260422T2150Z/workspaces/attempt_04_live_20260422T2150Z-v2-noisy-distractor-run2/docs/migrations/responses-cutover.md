# Responses Cutover

This workspace is cut over to the Responses wire path and event transcript mode.

- Set `wire_api = "responses"` and `transcript_mode = "responses_events"` in both `config/runtime.toml` and `.codex/config.toml`.
- Consume Responses output items as ordered events. Preserve wire order exactly instead of reconstructing order from rendered transcript text.
- Preserve tool-result correlation by carrying each `call_id` through normalization, replay serialization, replay loading, and transcript rendering.
- Keep replay event-sourced by serializing structured events directly. Do not rebuild state from rendered transcript text.
- Preserve the order of `output_text` blocks within each Responses message item so multi-block assistant output replays in the sequence it was emitted.
