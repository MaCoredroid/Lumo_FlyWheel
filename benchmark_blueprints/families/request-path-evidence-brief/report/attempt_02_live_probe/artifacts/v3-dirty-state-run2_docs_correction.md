# Docs Correction

Recommended correction for the support note:

`--owner` is parsed in `sync_app/cli.py::main` and passed directly to `sync_app/service.py::sync_item`. The service resolves the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner` before any record is created. It then creates the base record with `sync_app/store.py::make_record`, derives `routing_key` from the effective owner and item name in `sync_app/serializer.py::build_routing_key`, and emits the final payload in `sync_app/serializer.py::serialize_payload`.

What to remove from the old explanation:

- Do not say `owner_source` comes from storage. The stored record created by `sync_app/store.py::make_record` only contains `name`, `status`, and `owner`; `owner_source` is attached later by `sync_app/serializer.py::serialize_payload` after `sync_app/service.py::_resolve_owner` decides it.
- Do not say `routing_key` is computed before `--owner` is applied. `sync_app/service.py::sync_item` computes `routing_key` only after `sync_app/service.py::_resolve_owner` has already chosen the explicit or default owner.
- Do not route the fix into storage. The live path exercised by `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` shows the current behavior is already correct.

Short replacement text for docs:

The CLI passes the optional `--owner` value into the service. The service resolves both the effective owner and `owner_source`, creates the base record, derives `routing_key` from the resolved owner and item name, and then emits the payload. Storage is not the source of truth for `owner_source`, and `routing_key` is derived after owner resolution.
