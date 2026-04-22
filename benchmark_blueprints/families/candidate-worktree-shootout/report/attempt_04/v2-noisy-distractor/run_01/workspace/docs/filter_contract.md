# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels must be normalized by the shared service layer,
not by individual entrypoints. `report_filters.service.compile_filters`
is the contract owner for:

- trimming surrounding whitespace
- lowercasing labels
- converting repeated `-`, `_`, and `/` separators into single spaces
- collapsing repeated whitespace and dropping blank entries

The CLI stays thin and delegates to the shared service contract so the
scheduled importer and saved-view repair jobs receive the same canonical
keys.
