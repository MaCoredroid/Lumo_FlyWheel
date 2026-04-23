# Responses Cutover

Use the Responses wire path and keep the raw event stream as the replay source of truth.

- Set `config/runtime.toml` and `.codex/config.toml` to `wire_api = "responses"` and `transcript_mode = "responses_events"`.
- Read adapter payloads from Responses `output` items instead of the legacy chat-wrapper `choices[0].message` path.
- Preserve event ordering by sorting on `sequence` when it is present and otherwise keeping source order for unsequenced items.
- Preserve tool-result correlation by carrying each `call_id` through normalization, serialization, replay, and rendered transcript output.
- Expand `message` items from their `output_text` blocks in order, but keep replay event-sourced; rendered transcript text is a view and must not be used to reconstruct state.
- Pass through future or unknown event kinds without collapsing them so replay remains forward-compatible while keeping the original event order intact.
