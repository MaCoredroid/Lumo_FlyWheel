Inventory Ops Report needs a Markdown mode for its local report CLI.

The repo already supports JSON output, but the new tests expect
`--format markdown` to produce a readable heading, a summary line showing
how many sections and queued items are represented, and an owner table
that preserves every owner/label/count row returned by `build_sections()`.
Inventory Ops handoff now also needs an owner totals rollup in the
Markdown output so repeated owner rows are easy to scan during queue
handoff. The owner totals should be derived from the runtime sections,
sorted by queued items descending with owner name as the tie-breaker, and
the Markdown handoff should call out the busiest owner so the queue lead
can scan the summary before reading the detailed table.
Keep the JSON path working, add the Markdown renderer, and update
`docs/usage.md` so the documented commands and output description match
the code.

Do not remove or rewrite the existing tests.
