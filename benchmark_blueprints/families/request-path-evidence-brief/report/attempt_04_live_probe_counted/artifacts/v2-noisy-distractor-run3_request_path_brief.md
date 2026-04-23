# Request Path Evidence Brief

## Verdict

The support note is **not correct**.

- `owner_source` does **not** come from storage. The storage record built by `sync_app/store.py::make_record` contains only `name`, `status`, and `owner`. `owner_source` is derived earlier by `sync_app/service.py::_resolve_owner` and is appended only when `sync_app/serializer.py::serialize_payload` emits the payload.
- `routing_key` is **not** computed before the CLI applies `--owner`. The CLI accepts `--owner` in `sync_app/cli.py::main`, passes `args.owner` into `sync_app/service.py::sync_item`, and `sync_item` computes `routing_key` from the already-resolved `effective_owner` via `sync_app/serializer.py::build_routing_key`.

## Live Path

For the explicit `--owner` path, the request moves through these live symbols:

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item` with `owner=args.owner`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, which returns `(owner.strip(), "explicit")` when `--owner` is present.
3. `sync_app/service.py::sync_item` passes the resolved owner into `sync_app/store.py::make_record`, which stores only the base record fields.
4. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::build_routing_key` with `effective_owner` and `name`.
5. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::serialize_payload`, which copies the record and adds the already-derived `owner_source` and `routing_key`.

## Why The Support Note Was Plausible But Wrong

The confusion is understandable because the emitted payload already contains `owner`, and the current default owner in `config/defaults.json` is also `pm-oncall`, so an explicit `--owner pm-oncall` run and a default-owner run can emit the same `owner` and `routing_key` string. That output similarity does not change the code path: provenance comes from `sync_app/service.py::_resolve_owner`, not storage, and the `routing_key` call site remains after owner resolution in `sync_app/service.py::sync_item`.

## Test Evidence

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` asserts that `sync_item(..., owner="pm-oncall")` returns `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"`, matching the path through `sync_app/service.py::_resolve_owner`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` shows the alternate branch: when the flag is absent, `owner_source == "default"` and the owner comes from `config/defaults.json` through `sync_app/service.py::_load_defaults`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI hands `--owner` through `sync_app/cli.py::main` into the same emitted payload.

## Conclusion

The evidence points to a docs-only correction, not a storage-layer fix. The relevant live behavior is already consistent across `sync_app/cli.py::main`, `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.
