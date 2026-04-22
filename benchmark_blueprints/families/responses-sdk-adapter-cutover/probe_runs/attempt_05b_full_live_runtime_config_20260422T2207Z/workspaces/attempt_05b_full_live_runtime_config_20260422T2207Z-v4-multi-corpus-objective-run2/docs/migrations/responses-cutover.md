# Responses Cutover

Use the Responses wire path and store transcripts as ordered event records.

- Set `wire_api = "responses"` and `transcript_mode = "responses_events"` in `config/runtime.toml`.
- Normalize from `response.output` items, not `choices[0].message` wrapper content.
- Preserve event ordering from the source stream. When `sequence` is present, sort by it and keep the original index as the tie-breaker.
- Preserve tool-result correlation by carrying `call_id` through adapter normalization, replay serialization, and rendered output.
- Keep replay event-sourced. Reconstruct state from serialized events only; do not rebuild state from rendered transcript text.
- Treat rendered transcripts as a debug surface, not the source of truth for replay.
