# Docs Correction

## Corrected Support Guidance

The live implementation does not support the storage-layer explanation from `ops/support_note.md`. `--owner` is parsed in `sync_app/cli.py::main`, then `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` to determine both the effective owner and `owner_source`. Only after that does `sync_app/service.py::sync_item` call `sync_app/store.py::make_record` for the base record, `sync_app/serializer.py::build_routing_key` for `routing_key`, and `sync_app/serializer.py::serialize_payload` to emit the final payload.

## Suggested Replacement Wording

Replace the support-note theory with this statement:

The exported project-board payload derives `owner_source` in `sync_app/service.py::_resolve_owner`, not from storage. The stored record created by `sync_app/store.py::make_record` contains only `name`, `status`, and the already-resolved `owner`. `routing_key` is computed in `sync_app/serializer.py::build_routing_key` from that resolved owner and the item name, so it is computed after CLI owner precedence has been applied. The final payload is assembled by `sync_app/serializer.py::serialize_payload`.

## Why This Correction Matches Repo Evidence

`docs/data_flow.md` already warns that seeing `owner` in the stored record is not enough to prove where `owner_source` or `routing_key` are decided. `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all align with the traced service-first flow above.
