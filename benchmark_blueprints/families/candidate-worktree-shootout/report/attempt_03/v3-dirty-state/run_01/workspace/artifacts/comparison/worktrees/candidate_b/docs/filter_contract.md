# Filter Contract

Report filters are canonical keys used by the CLI, the scheduled
importer, and saved-view repair jobs.

Separator-heavy labels are normalized by `report_filters.service` so
every caller shares the same canonicalization contract. The CLI should
remain a thin adapter over that service behavior rather than owning its
own normalization rules.
