Inventory Ops Report needs a Markdown mode for its local report CLI.

The repo already supports JSON output, but the new tests expect
`--format markdown` to produce a readable heading, a summary line showing
how many sections and queued items are represented, and an owner table
that preserves every owner/label/count row returned by `build_sections()`.
Keep the JSON path working, add the Markdown renderer, and update
`docs/usage.md` so the documented commands and output description match
the code.

Do not remove or rewrite the existing tests.
