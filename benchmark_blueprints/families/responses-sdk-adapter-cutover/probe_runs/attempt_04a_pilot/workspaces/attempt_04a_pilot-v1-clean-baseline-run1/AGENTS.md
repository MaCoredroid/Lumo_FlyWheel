Migrate this workspace from the legacy chat-wrapper path to Responses event semantics.

Rules:
- Fix code, config, and docs in-place.
- Preserve event ordering and tool-result correlation.
- Keep replay event-sourced; do not rebuild state from rendered transcript text.
- Do not edit benchmark-owned tests or transcript fixtures.
