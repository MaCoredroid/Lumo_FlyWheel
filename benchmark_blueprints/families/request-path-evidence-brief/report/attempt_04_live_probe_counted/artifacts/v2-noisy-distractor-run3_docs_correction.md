# Docs Correction

## Corrected Support Note

The note in `ops/support_note.md` should be corrected to:

> `owner_source` is determined in `sync_app/service.py::_resolve_owner`, not in storage. `sync_app/store.py::make_record` stores only `name`, `status`, and `owner`. The emitted payload gets `owner_source` and `routing_key` later in `sync_app/serializer.py::serialize_payload`. `routing_key` is computed in `sync_app/serializer.py::build_routing_key` after `sync_app/service.py::sync_item` resolves the effective owner, so the CLI `--owner` flag is already applied when `routing_key` is derived. The fix belongs in docs, not in the storage layer.

## Evidence Behind The Correction

- `sync_app/cli.py::main` adds `--owner` to the parser and forwards `args.owner` into `sync_app/service.py::sync_item`.
- `sync_app/service.py::_resolve_owner` returns `"explicit"` when the owner argument is present and `"default"` when it falls back to `sync_app/service.py::_load_defaults`.
- `sync_app/store.py::make_record` does not accept or emit `owner_source` or `routing_key`; it only returns the base record.
- `sync_app/serializer.py::serialize_payload` is the emission point that appends `owner_source` and `routing_key`.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all match that flow.

## Rejected Alternatives

- Storage-layer fix: rejected because `sync_app/store.py::make_record` is not where `owner_source` or `routing_key` are produced.
- Reordering CLI application of `--owner`: rejected because `sync_app/cli.py::main` passes `args.owner` directly into `sync_app/service.py::sync_item` before the service computes `routing_key`.
