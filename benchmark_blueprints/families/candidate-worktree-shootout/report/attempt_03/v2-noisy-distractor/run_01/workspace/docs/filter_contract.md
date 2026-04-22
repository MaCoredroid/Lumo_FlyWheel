# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels must be normalized by the shared service layer,
not by a CLI-only hotfix. `service.compile_filters(...)` is the contract
owner for trimming, lowercasing, collapsing repeated whitespace, and
rewriting `-`, `_`, and `/` separators into spaces before the keys are
joined or stored.

`cli.render_filters(...)` remains a thin parser that splits raw input and
delegates normalization to `service.build_filter_query(...)`, so the CLI,
scheduled importer, and saved-view repair job all produce the same
canonical keys.
