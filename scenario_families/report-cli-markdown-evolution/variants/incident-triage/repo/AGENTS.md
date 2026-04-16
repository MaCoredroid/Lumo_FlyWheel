Incident Triage Report needs a Markdown mode for its local report CLI.

The repo already supports JSON output, but the new tests expect
`--format markdown` to produce a readable heading plus an owner table.
Keep the JSON path working, add the Markdown renderer, and update
`docs/usage.md` so the documented command matches the code.

Do not remove or rewrite the existing tests.
