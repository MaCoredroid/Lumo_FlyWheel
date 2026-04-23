# Request Path Evidence Brief

## Verdict

The support note is **not correct**.

- The claim that `owner_source` looks like it comes from storage is not supported by the live code path. Storage only produces a base record with `name`, `status`, and `owner` via `sync_app/store.py::make_record`. `owner_source` is derived in `sync_app/service.py::_resolve_owner` and only attached to the emitted payload in `sync_app/serializer.py::serialize_payload`.
- The claim that `routing_key` is probably computed before the CLI applies `--owner` is also not supported. The CLI forwards `args.owner` directly into `sync_app/service.py::sync_item` via `sync_app/cli.py::main`, and `sync_app/service.py::sync_item` computes `routing_key` from the already-resolved `effective_owner` using `sync_app/serializer.py::build_routing_key`.

## Live Path

1. `sync_app/cli.py::main` parses `--owner` and passes `args.owner` into `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, which returns both `effective_owner` and `owner_source`.
3. `sync_app/service.py::sync_item` passes `effective_owner` into `sync_app/store.py::make_record`, so storage receives only the resolved owner, not the derivation logic.
4. `sync_app/service.py::sync_item` builds `routing_key` from `effective_owner` and `name` through `sync_app/serializer.py::build_routing_key`.
5. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::serialize_payload`, which copies the stored record and appends `owner_source` and `routing_key` to the emitted payload.

## Field Decisions

- `owner_source`: decided in `sync_app/service.py::_resolve_owner` as `"explicit"` when a non-empty owner is provided, otherwise `"default"` from `config/defaults.json` through `sync_app/service.py::_load_defaults`.
- `routing_key`: decided in `sync_app/serializer.py::build_routing_key` from the resolved owner and item name after `sync_app/service.py::_resolve_owner` runs.
- Emission: the final payload is assembled in `sync_app/serializer.py::serialize_payload`, not in storage.

## Test Observations

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` expects `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"` when `sync_app/service.py::sync_item` receives `owner="pm-oncall"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` expects `owner_source == "default"` and `routing_key` built from the default owner in `config/defaults.json` when the CLI flag is absent.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path preserves the explicit owner through to emitted `owner_source` and `routing_key`.

## Conclusion

Repo-local evidence points to a documentation misunderstanding, not a storage-layer bug. The smallest accurate correction is to state that storage only carries the resolved `owner`, while `owner_source` and `routing_key` are derived and emitted after owner resolution in the service/serializer path: `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.
