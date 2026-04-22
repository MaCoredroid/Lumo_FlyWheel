# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels are normalized by the shared service contract in
`service.compile_filters(...)`, which canonicalizes casing, whitespace,
and separator runs before any caller persists or compares report keys.

`cli.render_filters(...)` remains a thin entrypoint that forwards raw
comma-separated pieces into the service layer. Scheduled importer and
saved-view repair callers must rely on the same service normalization
rather than reimplementing CLI-local cleanup.
