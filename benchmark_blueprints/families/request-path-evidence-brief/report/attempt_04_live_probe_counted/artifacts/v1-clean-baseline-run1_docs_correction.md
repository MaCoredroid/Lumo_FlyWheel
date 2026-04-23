# Docs Correction

## Support Note Correction

The support note should be corrected to say that the exported payload is assembled in the service/serializer path, not in storage. `sync_app/cli.py::main` forwards `--owner` into `sync_app/service.py::sync_item`. `sync_app/service.py::_resolve_owner` decides both the effective owner and the `owner_source` label. `sync_app/store.py::make_record` persists only the base record fields (`name`, `status`, `owner`). `sync_app/serializer.py::build_routing_key` derives `routing_key` from the resolved owner and item name, and `sync_app/serializer.py::serialize_payload` emits both `owner_source` and `routing_key`.

## Recommended Replacement Text

Replace the support note with:

> `owner_source` is decided in `sync_app/service.py::_resolve_owner`, not recovered from storage. `routing_key` is derived in `sync_app/serializer.py::build_routing_key` after `sync_app/cli.py::main` has already forwarded `--owner` into `sync_app/service.py::sync_item`. `sync_app/store.py::make_record` stores only the base record, and `sync_app/serializer.py::serialize_payload` adds `owner_source` and `routing_key` to the exported payload. Based on the current repo, this is a docs clarification, not a storage-layer fix.

## Why This Fix Scope Is Enough

`tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` already pin the active behavior. The correction is to the explanation, not to the implementation.
