# Docs Correction

## Correction To The Support Note

`ops/support_note.md` should be corrected to say:

- `owner_source` is decided in `sync_app/service.py::_resolve_owner`, before serialization and independently of storage.
- `routing_key` is computed in `sync_app/serializer.py::build_routing_key` after `sync_app/service.py::_resolve_owner` chooses the effective owner.
- `sync_app/store.py::make_record` stores only the base record fields and is not the source of `owner_source` or `routing_key`.
- The appropriate action is docs cleanup, not a storage-layer behavior change.

## Evidence Basis

- `sync_app/cli.py::main` is the only live CLI entrypoint for `--owner`.
- `sync_app/service.py::sync_item` shows the live ordering: resolve owner, persist record, build routing key, serialize payload.
- `sync_app/serializer.py::serialize_payload` appends `owner_source` and `routing_key` to a copied record; it does not infer them from storage.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` and `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` encode the implemented behavior.

## Suggested Replacement Text

Suggested replacement for the historical support note:

> The live path accepts `--owner` in `sync_app/cli.py::main`, resolves `effective_owner` and `owner_source` in `sync_app/service.py::_resolve_owner`, persists the base record with `sync_app/store.py::make_record`, then computes `routing_key` with `sync_app/serializer.py::build_routing_key` and emits the final payload with `sync_app/serializer.py::serialize_payload`. Storage is not the source of `owner_source`, and `routing_key` is not computed before owner resolution.
