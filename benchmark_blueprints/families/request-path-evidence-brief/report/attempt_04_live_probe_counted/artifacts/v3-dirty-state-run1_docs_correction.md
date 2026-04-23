# Docs Correction

## Support Note Correction

`ops/support_note.md` should be corrected to say:

- `owner_source` does **not** come from storage. The live record written by `sync_app/store.py::make_record` contains only `name`, `status`, and `owner`. `owner_source` is derived in `sync_app/service.py::_resolve_owner` and added to the emitted payload in `sync_app/serializer.py::serialize_payload`.
- `routing_key` is **not** computed before the CLI applies `--owner`. `sync_app/cli.py::main` forwards `args.owner` into `sync_app/service.py::sync_item`, which resolves the effective owner first and only then calls `sync_app/serializer.py::build_routing_key`.
- No storage-layer fix is indicated by the current repo evidence. This is a documentation correction against `ops/support_note.md`, not a behavior change request.

## Suggested Replacement Text

Suggested replacement for the support note:

> The emitted payload does not derive `owner_source` from storage. The CLI forwards `--owner` into the service, `sync_item` resolves the effective owner and `owner_source`, stores the base record with `owner`, computes `routing_key` from the resolved owner and item name, and then emits both derived fields with the payload.

## Evidence Anchors

- `sync_app/cli.py::main`
- `sync_app/service.py::sync_item`
- `sync_app/service.py::_resolve_owner`
- `sync_app/store.py::make_record`
- `sync_app/serializer.py::build_routing_key`
- `sync_app/serializer.py::serialize_payload`
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`
