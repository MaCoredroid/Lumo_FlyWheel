# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels are normalized by the shared service layer.
`service.compile_filters(...)` is the contract owner for trimming,
lowercasing, collapsing separator runs, and dropping blank entries.

The CLI passes raw comma-split labels into the service and must not
carry its own normalization branch. Direct service callers rely on the
same behavior, so any future normalization change belongs in the service
contract instead of `cli.py`.
