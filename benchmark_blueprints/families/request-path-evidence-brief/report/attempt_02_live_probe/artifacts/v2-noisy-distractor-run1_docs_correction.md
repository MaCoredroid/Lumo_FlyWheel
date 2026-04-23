# Docs Correction

## Support Note Correction

The current support note in `ops/support_note.md` should be corrected to say:

- `owner_source` is derived in `sync_app/service.py::_resolve_owner`, not from storage.
- `routing_key` is derived in `sync_app/serializer.py::build_routing_key` from the resolved owner and item name after `sync_app/cli.py::main` passes `--owner` into `sync_app/service.py::sync_item`.
- The stored record built by `sync_app/store.py::make_record` contains `name`, `status`, and `owner`, but does not persist `owner_source` or `routing_key`.
- The issue is documentation accuracy, not live storage behavior.

## Evidence Summary

- `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item` with `owner=args.owner`.
- `sync_app/service.py::_resolve_owner` returns `"explicit"` for a non-empty CLI owner and `"default"` when the flag is absent.
- `sync_app/store.py::make_record` returns only `{"name": ..., "status": ..., "owner": ...}`.
- `sync_app/serializer.py::serialize_payload` appends `owner_source` and `routing_key` to the emitted payload.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` and `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` confirm the live behavior.

## Proposed Replacement Text

Suggested replacement for the support note:

> The live path resolves the effective owner in the service layer, stores the base record with that owner, and then emits `owner_source` and `routing_key` in the serializer. `owner_source` is not read back from storage, and `routing_key` is derived after the CLI-supplied `--owner` has already been applied. See `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, and `sync_app/serializer.py::serialize_payload`.
