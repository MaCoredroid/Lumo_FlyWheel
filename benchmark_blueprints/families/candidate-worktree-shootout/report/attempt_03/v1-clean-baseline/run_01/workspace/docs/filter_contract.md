# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels must be normalized by the service layer so every
caller shares one contract, including direct service users that bypass
the CLI.
