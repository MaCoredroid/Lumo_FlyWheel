# Docs Correction Note

## Scope

This note does not change repo behavior. It records doc-level corrections supported by local code and tests.

## Confirmed Accurate

- `docs/cli.md` is directionally correct that the payload includes `owner`, `owner_source`, and `routing_key`, and that omitting `--owner` falls back to `config/defaults.json`. That matches `sync_app/cli.py::main`, `sync_app/service.py::_resolve_owner`, and `sync_app/serializer.py::serialize_payload`.
- `docs/data_flow.md` is directionally correct that the CLI calls the service and that the record alone does not prove the derivation point for `owner_source` or `routing_key`. That matches `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, and `sync_app/serializer.py::serialize_payload`.

## Corrections Needed

### 1. Reject the storage-origin explanation for `owner_source`

`ops/support_note.md` says `owner_source` looks like it comes from storage. The live code shows otherwise: `sync_app/store.py::make_record` stores only `name`, `status`, and `owner`, while `owner_source` is derived in `sync_app/service.py::_resolve_owner` and attached later in `sync_app/serializer.py::serialize_payload`.

Recommended correction:

> `owner_source` is derived in the service during owner resolution, not loaded from storage.

### 2. Reject the claim that `routing_key` is computed before CLI owner application

`ops/support_note.md` says `routing_key` is probably computed before the CLI applies `--owner`. The live path shows the opposite: `sync_app/cli.py::main` passes `args.owner` into `sync_app/service.py::sync_item`, and `sync_app/service.py::sync_item` computes `routing_key` only after `_resolve_owner` returns `effective_owner`. The derivation happens in `sync_app/serializer.py::build_routing_key`.

Recommended correction:

> `routing_key` is computed in the service from the resolved effective owner and the request name, after CLI argument parsing has already supplied the optional `--owner` value.

### 3. Make the live ordering explicit in docs

`docs/data_flow.md` is correct but compressed. A clearer version would name the exact order:

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item`.
2. `sync_app/service.py::_resolve_owner` derives `effective_owner` and `owner_source`.
3. `sync_app/store.py::make_record` builds the base record.
4. `sync_app/serializer.py::build_routing_key` derives `routing_key`.
5. `sync_app/serializer.py::serialize_payload` emits the final payload.

## Supporting Tests

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`
- `tests/test_docs.py::test_docs_reference_owner_path_surfaces`
