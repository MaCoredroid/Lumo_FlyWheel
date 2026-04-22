# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels must be normalized by
`service.compile_filters(...)`, which is the shared contract for the
CLI, the scheduled importer, and saved-view repair jobs.

`cli.render_filters(...)` remains a thin entrypoint that only splits the
raw comma-delimited input before delegating to the service layer.

Canonical normalization replaces runs of `-`, `_`, and `/` with spaces,
collapses internal whitespace, lowercases the label, and drops blank
entries after normalization.
