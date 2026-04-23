# Docs Correction

No repo behavior change is indicated by the trace. If anything is corrected, it should be the support note wording, not the storage layer description.

Suggested replacement for the support note:

`--owner` is parsed in `sync_app/cli.py::main` and forwarded to `sync_app/service.py::sync_item`. The service resolves both the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner`, persists only the base record via `sync_app/store.py::make_record`, computes `routing_key` from the resolved owner in `sync_app/serializer.py::build_routing_key`, and emits `owner`, `owner_source`, and `routing_key` in `sync_app/serializer.py::serialize_payload`.

Why this correction is needed:

- The live storage helper `sync_app/store.py::make_record` does not write `owner_source`, so storage is not the source of that emitted field.
- `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` before `sync_app/serializer.py::build_routing_key`, so `routing_key` is derived after the CLI owner override has been applied.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all assert the resolved-owner behavior above.

Current docs status:

`docs/data_flow.md` already matches the live path described by `sync_app/service.py::sync_item` and does not need a behavioral correction based on this repo snapshot.
