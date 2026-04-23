# Docs Correction

The current support note should be corrected to match the implemented path.

## Corrected Statement

- `--owner` is accepted at `sync_app/cli.py::main` and forwarded into `sync_app/service.py::sync_item`.
- The effective owner and `owner_source` are decided in `sync_app/service.py::_resolve_owner`.
- The base stored record is created by `sync_app/store.py::make_record`; it does not establish `owner_source`.
- `routing_key` is derived in `sync_app/serializer.py::build_routing_key` from the already resolved owner and item name.
- The final emitted payload is assembled in `sync_app/serializer.py::serialize_payload`, which adds both `owner_source` and `routing_key` to the copied record.

## Recommended Replacement For The Support Note

Replace the current note with:

> `--owner` flows from `sync_app/cli.py::main` into `sync_app/service.py::sync_item`. The service resolves the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner`, creates the base record via `sync_app/store.py::make_record`, computes `routing_key` via `sync_app/serializer.py::build_routing_key`, and emits the final payload through `sync_app/serializer.py::serialize_payload`. This is a docs clarification, not a storage-layer bug.

## Why This Correction Is Needed

- `sync_app/store.py::make_record` only returns `name`, `status`, and `owner`, so it cannot be the source of `owner_source`.
- `sync_app/service.py::sync_item` computes `routing_key` after owner resolution, so the note’s "before the CLI applies `--owner`" suspicion is not consistent with the implementation.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` and `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` already lock in the intended behavior.
