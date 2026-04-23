# Request Path Evidence Brief

## Verdict
The support note is **not correct**. `owner_source` does not come from storage, and `routing_key` is not computed before the CLI-applied `--owner` reaches the service. The live code path shows that `sync_app/cli.py::main` forwards `args.owner` into `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner` derives both the effective owner and `owner_source`, `sync_app/store.py::make_record` stores only the resolved `owner`, `sync_app/serializer.py::build_routing_key` computes `routing_key` from that resolved owner, and `sync_app/serializer.py::serialize_payload` appends both `owner_source` and `routing_key` after the record is created.

## Live Path
- `sync_app/cli.py::main` parses `--owner` and passes `args.owner` directly to `sync_app/service.py::sync_item`.
- `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` first, so owner selection happens before any record creation or payload serialization.
- `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` for a present flag and falls back to `config/defaults.json` through `sync_app/service.py::_load_defaults` with `"default"` otherwise.
- `sync_app/store.py::make_record` receives only `name`, `status`, and the resolved `owner`; it does not compute or persist `owner_source`.
- `sync_app/serializer.py::build_routing_key` is called from `sync_app/service.py::sync_item` with `effective_owner`, so `routing_key` is derived after owner resolution, not before CLI input is applied.
- `sync_app/serializer.py::serialize_payload` copies the stored record and adds `owner_source` plus `routing_key` at emission time.

## Field Derivations
- `owner_source` is derived in `sync_app/service.py::_resolve_owner`, not read from `sync_app/store.py::make_record` output.
- `routing_key` is derived in `sync_app/serializer.py::build_routing_key` from the resolved owner and item name supplied by `sync_app/service.py::sync_item`.
- The emitted payload is assembled in `sync_app/serializer.py::serialize_payload`; the record is only the base input to that emission step.

## Rejected Decoys
- `sync_app/serializer.py::draft_owner_source_from_record` is not on the live call path. No production code or test calls it.
- `sync_app/store.py::legacy_make_record_with_owner_source` is not on the live call path. The live service imports and calls `sync_app/store.py::make_record` instead.
- `sync_app/store.py::legacy_build_routing_key` is not on the live call path. The live service imports and calls `sync_app/serializer.py::build_routing_key` instead.

## Test Observations
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` confirms the explicit-owner path emits `owner="pm-oncall"`, `owner_source="explicit"`, and `routing_key="pm-oncall:launch-checklist"`, which matches the live sequence through `sync_app/service.py::sync_item`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` confirms the fallback path uses `config/defaults.json` through `sync_app/service.py::_load_defaults` and emits `owner_source="default"`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms `sync_app/cli.py::main` passes `--owner` through to the service and the final payload still includes the explicit owner-derived fields.
- `tests/test_docs.py::test_docs_reference_owner_path_surfaces` confirms the checked-in docs mention `--owner`, `owner_source`, `routing_key`, and `sync_item`, but it does not validate the incorrect inference in `ops/support_note.md`.

## Fix Scope
The evidence supports a docs correction only. Nothing in the live code suggests a storage-layer fix. The misleading part is the interpretation in `ops/support_note.md`, not the behavior implemented by `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, and `sync_app/serializer.py::serialize_payload`.
