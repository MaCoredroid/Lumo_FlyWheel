# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy normalization is owned by
`service.compile_filters(...)`. Callers may pass raw labels containing
mixed case, repeated whitespace, and separator runs such as `-`, `_`,
and `/`; the service canonicalizes them into lowercase space-separated
keys before blank entries are dropped.

`cli.render_filters(...)` is only an input splitter. Direct callers that
bypass the CLI still receive the same normalized filter keys because the
service layer is the shared contract authority.
