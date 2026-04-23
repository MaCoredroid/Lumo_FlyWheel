# Request Path Evidence Brief

## Verdict

The support note is **not correct**.

- `--owner` is parsed in `sync_app/cli.py::main` and forwarded directly into `sync_app/service.py::sync_item`.
- `owner_source` is decided in `sync_app/service.py::_resolve_owner`, not inferred from storage. The stored base record is created by `sync_app/store.py::make_record`, which only returns `name`, `status`, and `owner`.
- `routing_key` is computed inside `sync_app/service.py::sync_item` after the effective owner has already been resolved, using `sync_app/serializer.py::build_routing_key`.
- The final emitted payload gets `owner_source` and `routing_key` attached in `sync_app/serializer.py::serialize_payload`.

## Live Path

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item` with `owner=args.owner`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` first.
3. `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when `--owner` is present, otherwise it falls back through `sync_app/service.py::_load_defaults`.
4. `sync_app/service.py::sync_item` then builds the base record with `sync_app/store.py::make_record`.
5. `sync_app/service.py::sync_item` computes `routing_key` by calling `sync_app/serializer.py::build_routing_key(effective_owner, name)`.
6. `sync_app/service.py::sync_item` emits the final payload via `sync_app/serializer.py::serialize_payload(record, owner_source=owner_source, routing_key=routing_key)`.

## Why The Support Note Fails

- The note says `owner_source` "looks like it comes from storage." That is disproved by `sync_app/store.py::make_record`, which does not produce `owner_source`, and by `sync_app/serializer.py::serialize_payload`, which injects `owner_source` during emission.
- The note says `routing_key` is "probably computed before the CLI applies `--owner`." That is disproved by `sync_app/cli.py::main`, which passes `args.owner` into `sync_app/service.py::sync_item`, and by `sync_app/service.py::sync_item`, which computes `routing_key` only after `sync_app/service.py::_resolve_owner` returns the effective owner.
- The note asks whether the fix belongs in storage. Repo-local evidence points to a docs correction only; the active request path already matches the intended behavior in `sync_app/service.py::sync_item`.

## Test-Backed Observations

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` confirms explicit `--owner` yields `owner="pm-oncall"`, `owner_source="explicit"`, and `routing_key="pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` confirms the fallback path comes from `sync_app/service.py::_load_defaults` and still derives `routing_key` from the resolved owner.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path preserves the same resolved values end to end.

## Rejected Decoys

- `sync_app/serializer.py::draft_owner_source_from_record` is not used by the live path and therefore does not prove `owner_source` comes from storage.
- `sync_app/store.py::legacy_make_record_with_owner_source` is unused legacy code and not part of the active request path.
- `sync_app/store.py::legacy_build_routing_key` is unused legacy code and not part of the active request path.
