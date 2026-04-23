# Docs Correction

## Recommended correction

Replace the support-note interpretation with the following:

The effective owner is resolved in `sync_app/service.py::_resolve_owner`, not in storage. `sync_app/store.py::make_record` stores only `name`, `status`, and the resolved `owner`. After that, `sync_app/serializer.py::build_routing_key` derives `routing_key` from the already-resolved owner and item name, and `sync_app/serializer.py::serialize_payload` appends both `owner_source` and `routing_key` to the emitted payload. The CLI only parses `--owner` and forwards it through `sync_app/cli.py::main` into `sync_app/service.py::sync_item`.

## Scope of the fix

This should be treated as a docs/support-note correction only. Repo-local evidence does not support a storage-layer fix:

- `sync_app/service.py::_resolve_owner` is the only live derivation point for `owner_source`.
- `sync_app/store.py::make_record` does not persist `owner_source` or `routing_key`.
- `sync_app/service.py::sync_item` computes `routing_key` only after owner resolution.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirm that explicit `--owner` input controls both `owner_source` and `routing_key`.

## Suggested wording for future docs

The CLI forwards optional `--owner` input into the service. The service resolves `(effective_owner, owner_source)`, stores a base record with `owner`, derives `routing_key` from the resolved owner and name, and emits the final payload by adding `owner_source` and `routing_key` after record creation.
