# Request Path Evidence Brief

## Verdict

The support note is **not correct**. Repo-local evidence shows that `--owner` is parsed in `sync_app/cli.py::main`, applied before payload assembly in `sync_app/service.py::sync_item`, resolved into `owner_source` by `sync_app/service.py::_resolve_owner`, and then used to derive `routing_key` in `sync_app/serializer.py::build_routing_key`. The storage helper `sync_app/store.py::make_record` only returns `name`, `status`, and `owner`; it does not derive `owner_source` or `routing_key`.

## Live Path

1. `sync_app/cli.py::main` accepts `--owner` and forwards `args.owner` into `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` first, so the effective owner is decided before any record or payload is built.
3. `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when `--owner` is present, otherwise it falls back to `config/defaults.json` via `sync_app/service.py::_load_defaults`.
4. `sync_app/service.py::sync_item` passes the effective owner into `sync_app/store.py::make_record`, which only builds the base record with `owner`.
5. `sync_app/service.py::sync_item` computes `routing_key` from the same effective owner by calling `sync_app/serializer.py::build_routing_key`.
6. `sync_app/serializer.py::serialize_payload` emits the final payload by copying the record and appending `owner_source` and `routing_key`.

## Field Findings

- `owner_source` is not inferred from storage. It is derived in `sync_app/service.py::_resolve_owner` and attached during emission in `sync_app/serializer.py::serialize_payload`.
- `routing_key` is not computed before the CLI applies `--owner`. The CLI passes the flag into `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key` runs afterward from `effective_owner`.
- The current repo evidence supports a docs correction, not a storage-layer fix. `ops/support_note.md` conflicts with the implemented flow, while `docs/data_flow.md` already warns that stored `owner` alone is insufficient to prove where `owner_source` or `routing_key` are decided.

## Test-Backed Observations

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` expects `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"` when `sync_app/service.py::sync_item` receives `owner="pm-oncall"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` expects `owner_source == "default"` and the routing key to use the default owner from `config/defaults.json`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path preserves the explicit owner through to the emitted payload.
- Running `python -m sync_app.cli --name 'Launch Checklist' --status pending --owner pm-oncall` emitted JSON with `owner: "pm-oncall"`, `owner_source: "explicit"`, and `routing_key: "pm-oncall:launch-checklist"`, matching the code path above.

## Rejected Decoys

- `sync_app/serializer.py::draft_owner_source_from_record` is not part of the live request path. Nothing in the repo calls it.
- `sync_app/store.py::legacy_make_record_with_owner_source` is a legacy helper, not the live path used by `sync_app/service.py::sync_item`.
- `sync_app/store.py::legacy_build_routing_key` is also unused in the live path; `sync_app/service.py::sync_item` imports and calls `sync_app/serializer.py::build_routing_key` instead.
