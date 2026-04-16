`inventory-ops` is the richer small-investigative benchmark for the
`report-cli-markdown-evolution` family.

The hidden tests are split so the baseline Markdown feature and the dormant
watchlist follow-up can be reasoned about separately:

- round-1 hidden slices cover the CLI, Markdown layout, docs, differential
  fixtures, and shared sorting invariants
- follow-up slices cover the dormant `include_known_owners` flag and the
  shared zero-count wording used once that flag is enabled

The red-team patches exercise the expected shortcuts for this repo shape:
hardcoded Markdown, fake pytest shims, and markdown-only watchlist fixes that
avoid the shared summary layer.
