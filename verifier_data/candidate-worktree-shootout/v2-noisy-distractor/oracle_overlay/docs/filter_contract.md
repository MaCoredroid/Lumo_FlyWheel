# Filter Contract

Normalization ownership lives in `service.compile_filters(...)`.

All callers hand raw labels to the shared service layer, including:

- the CLI path
- the scheduled importer
- the saved-view repair job

`cli.py` should stay thin and must not duplicate separator cleanup that
direct service callers also rely on.
