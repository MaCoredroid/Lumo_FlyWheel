# Docs Correction

Corrected statement for the request path:

`sync_app/cli.py::main` parses `--owner` and passes it into `sync_app/service.py::sync_item`. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` to derive both the effective owner and `owner_source`, writes the base record through `sync_app/store.py::make_record`, computes `routing_key` via `sync_app/serializer.py::build_routing_key`, and emits the final payload through `sync_app/serializer.py::serialize_payload`. The stored base record contains `owner`, but `owner_source` and `routing_key` are derived in the service/serializer path, not loaded from storage.

Support-note correction:

- Replace “`owner_source` in the exported payload looks like it comes from storage” with “`owner_source` is derived in `sync_app/service.py::_resolve_owner` before payload serialization.”
- Replace “`routing_key` is probably computed before the CLI applies `--owner`” with “`sync_app/cli.py::main` passes `--owner` into `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key` runs afterward on the resolved owner.”
- Replace “the fix may belong in the storage layer” with “repo-local evidence points to a docs/support-note correction only; `sync_app/store.py::make_record` stores the resolved `owner`, while the legacy storage helpers are not on the live path.”

Why the old note drifted:

The shortcut came from observing that `sync_app/store.py::make_record` returns a record that already contains `owner`. That observation is true but incomplete. It does not show where `owner_source` is decided, and it does not show when `routing_key` is computed. The live call chain in `sync_app/service.py::sync_item` establishes both facts directly, and the expectations in `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` match that chain.
