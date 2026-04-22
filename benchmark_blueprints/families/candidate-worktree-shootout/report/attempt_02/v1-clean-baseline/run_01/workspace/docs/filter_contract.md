# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels must be normalized in the service layer before
filters are returned or joined into query strings. The CLI remains a thin
parser that forwards raw comma-delimited labels into the shared service
contract.

This service-owned normalization is required because direct callers can
bypass the CLI. Importers, repair jobs, and other service consumers must
receive the same canonical keys as the interactive CLI path.
