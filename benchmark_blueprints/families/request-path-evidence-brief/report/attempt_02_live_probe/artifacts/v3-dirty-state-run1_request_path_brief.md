# Request Path Evidence Brief

## Verdict

The support note is **not correct**.

- `owner_source` is not derived from storage in the live path. The stored record created by `sync_app/store.py::make_record` contains only `name`, `status`, and `owner`, while `owner_source` is derived earlier by `sync_app/service.py::_resolve_owner` and appended later by `sync_app/serializer.py::serialize_payload`.
- `routing_key` is not computed before the CLI applies `--owner`. The CLI forwards `args.owner` into `sync_app/service.py::sync_item` via `sync_app/cli.py::main`, and `sync_item` computes `routing_key` only after the effective owner has been resolved.

## Live Path

1. `sync_app/cli.py::main` parses `--owner` and passes `args.owner` into `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, which returns both the effective owner and `owner_source`.
3. `sync_app/service.py::sync_item` persists the base record through `sync_app/store.py::make_record`; that record carries `owner` but does not carry `owner_source` or `routing_key`.
4. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::build_routing_key` with the resolved owner and the item name.
5. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::serialize_payload`, which copies the base record and appends `owner_source` and `routing_key` to the emitted payload.

## Evidence

- `sync_app/cli.py::main` defines `--owner` and forwards it as `owner=args.owner` into `sync_app/service.py::sync_item`.
- `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` for a non-empty owner and otherwise returns the default owner with `"default"`.
- `sync_app/service.py::_load_defaults` provides the fallback owner by reading `config/defaults.json`.
- `sync_app/store.py::make_record` proves storage is only the base record layer because it returns `{"name": name, "status": status, "owner": owner}` and nothing else.
- `sync_app/serializer.py::build_routing_key` proves the live routing key is derived from the resolved owner and name, not from the CLI before resolution and not from storage.
- `sync_app/serializer.py::serialize_payload` proves emission is where `owner_source` and `routing_key` are added to the output payload.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`, and `tests/test_docs.py::test_docs_reference_owner_path_surfaces` all match this code path.

## Rejected Decoys

- `sync_app/store.py::legacy_make_record_with_owner_source` is not part of the live path. Repo search shows no live caller for it.
- `sync_app/store.py::legacy_build_routing_key` is not part of the live path. `sync_app/service.py::sync_item` imports and calls `sync_app/serializer.py::build_routing_key` instead.
- `sync_app/serializer.py::draft_owner_source_from_record` is not part of the live path. Repo search shows no caller.

## Scope Of Fix

Based on the current repo state, the issue is documentation accuracy in the support note, not storage behavior. The live behavior is established by `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, `sync_app/serializer.py::serialize_payload`, and confirmed by `tests/test_docs.py::test_docs_reference_owner_path_surfaces`, so the correction belongs in the stale support narrative rather than in repo code.
