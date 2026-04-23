# Request Path Brief

## Verdict

The support note is **not correct**. Repo-local evidence shows that `--owner` is parsed in `sync_app/cli.py::main`, passed into `sync_app/service.py::sync_item`, resolved by `sync_app/service.py::_resolve_owner`, and only then used to build both the stored record and the emitted `routing_key`. `owner_source` is not loaded from storage on the live path; it is derived in the service layer and attached during payload serialization in `sync_app/serializer.py::serialize_payload`.

## Live Path

The request path is:

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, which returns `(owner.strip(), "explicit")` when the flag is present and otherwise returns the default owner plus `"default"`.
3. `sync_app/service.py::sync_item` passes the resolved owner into `sync_app/store.py::make_record`, so storage receives only `name`, `status`, and the already-resolved `owner`.
4. `sync_app/service.py::sync_item` computes `routing_key` by calling `sync_app/serializer.py::build_routing_key` with the same resolved owner and the item name.
5. `sync_app/service.py::sync_item` emits the final payload through `sync_app/serializer.py::serialize_payload`, which copies the record and appends `owner_source` and `routing_key`.

This matches the implementation evidence in `sync_app/store.py::make_record` and `sync_app/serializer.py::serialize_payload`: the stored record stops at `owner`, while `owner_source` and `routing_key` are attached only during emission.

## Why The Support Note Fails

- The claim that `owner_source` "looks like it comes from storage" is contradicted by `sync_app/store.py::make_record`, which stores only `name`, `status`, and `owner`, and by `sync_app/serializer.py::serialize_payload`, which appends `owner_source` after the record is created.
- The claim that `routing_key` is "probably computed before the CLI applies `--owner`" is contradicted by `sync_app/cli.py::main`, which passes `args.owner` directly into `sync_app/service.py::sync_item`, and by `sync_app/service.py::sync_item`, which computes `routing_key` from `effective_owner` after `_resolve_owner`.

## Evidence From Tests And Runtime

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` asserts that `sync_item(..., owner="pm-oncall")` emits `"owner_source": "explicit"` and `"routing_key": "pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` asserts the default-owner path emits `"owner_source": "default"` and derives `routing_key` from the default owner in `config/defaults.json`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path preserves the explicit owner through emission.
- Running `python -m sync_app.cli --name "Launch Checklist" --status pending --owner pm-oncall` emitted `owner_source="explicit"` and `routing_key="pm-oncall:launch-checklist"`, matching the service-layer derivation described above.

## Storage-Layer Vs Docs-Layer Fix

The repo evidence supports a docs-only correction, not a storage-layer behavior fix. The live implementation and tests already agree on the path through `sync_app/cli.py::main`, `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.
