Migrate this workspace from the legacy chat-wrapper path to Responses event semantics.

Rules:
- Fix code, config, and docs in-place.
- The required config file is `.codex/config.toml`.
- The required migration note is `docs/migrations/responses-cutover.md`.
- Set the config to the Responses wire path and Responses event transcript mode.
- Update the migration note so it explicitly covers event ordering and tool-result correlation.
- Preserve event ordering and tool-result correlation.
- Keep replay event-sourced; do not rebuild state from rendered transcript text.
- Do not edit benchmark-owned tests or transcript fixtures.
