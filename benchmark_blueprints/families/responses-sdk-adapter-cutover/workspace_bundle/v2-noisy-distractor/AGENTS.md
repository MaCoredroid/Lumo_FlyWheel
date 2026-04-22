Migrate this workspace from the legacy chat-wrapper path to Responses event semantics.

Rules:
- Fix code, config, and docs in-place.
- The required config file is `config/runtime.toml`.
- The required migration note is `docs/migrations/responses-cutover.md`.
- Set the config to the Responses wire path and Responses event transcript mode.
- `config/runtime.toml` lives in the workspace root; update that file directly.
- Update the migration note so it explicitly covers event ordering and tool-result correlation.
- Use workspace-relative paths when editing files in this benchmark workspace.
- Preserve event ordering and tool-result correlation.
- Keep replay event-sourced; do not rebuild state from rendered transcript text.
- Do not edit benchmark-owned tests or transcript fixtures.
