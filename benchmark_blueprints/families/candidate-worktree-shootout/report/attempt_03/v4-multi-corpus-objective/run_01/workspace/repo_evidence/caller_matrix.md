# Caller Matrix

Current report-filter callers:

- `cli.render_filters(...)` -> `service.build_filter_query(...)`
- scheduled importer -> `service.compile_filters(...)`
- saved-view repair job -> `service.compile_filters(...)`

Any fix that only touches `cli.py` leaves at least two direct callers
broken.
