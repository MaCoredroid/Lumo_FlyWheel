Incident Triage Report needs a Markdown mode for its local report CLI.

The repo already supports JSON output, but the new tests expect
`--format markdown` to render the incident triage summary with the same
runtime data:

- a title heading
- a one-line incident summary showing active incidents and ack SLA breaches
- a "Highest load owner" callout derived from the runtime summary
- an "Active Queue" table that preserves every runtime incident row
- an "Owner Load" table that rolls incidents up by owner

Owner load should be sorted by active incident count descending, then by ack
SLA breaches descending, then owner name ascending. Keep the JSON path
working, add the Markdown renderer, and update `docs/usage.md` so the
documented commands and Markdown sections match the code.

Do not remove or rewrite the existing tests.
