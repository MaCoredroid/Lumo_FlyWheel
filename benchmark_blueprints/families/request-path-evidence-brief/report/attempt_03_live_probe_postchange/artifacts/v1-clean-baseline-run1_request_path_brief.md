# Request Path Evidence Brief

## Verdict

The support note is **not correct**. Repo-local evidence shows that `owner_source` is not read from storage, and `routing_key` is not computed before the CLI-applied `--owner` reaches the service. The live path is `sync_app/cli.py::main` -> `sync_app/service.py::sync_item` -> `sync_app/service.py::_resolve_owner` / `sync_app/store.py::make_record` / `sync_app/serializer.py::build_routing_key` -> `sync_app/serializer.py::serialize_payload`.

The storage layer only returns the base record fields `name`, `status`, and `owner` through `sync_app/store.py::make_record`. The emitted `owner_source` and `routing_key` fields are added later by `sync_app/serializer.py::serialize_payload`, using values produced in `sync_app/service.py::_resolve_owner` and `sync_app/serializer.py::build_routing_key`.

## Request Path

`sync_app/cli.py::main` parses `--owner` and passes `args.owner` directly into `sync_app/service.py::sync_item`.

`sync_app/service.py::sync_item` resolves the effective owner first by calling `sync_app/service.py::_resolve_owner`. That helper returns `(owner.strip(), "explicit")` when `--owner` is present, otherwise it loads the fallback owner from `config/defaults.json` and returns `"default"`.

After owner resolution, `sync_app/service.py::sync_item` calls `sync_app/store.py::make_record` with the already-resolved owner, then separately calls `sync_app/serializer.py::build_routing_key` with that same effective owner and the item name, and finally emits the payload through `sync_app/serializer.py::serialize_payload`.

## Field Findings

`owner_source` is derived in `sync_app/service.py::_resolve_owner`, not in storage. `sync_app/store.py::make_record` has no `owner_source` parameter, while `sync_app/serializer.py::serialize_payload` appends the already-derived `owner_source` to the outgoing payload.

`routing_key` is derived in `sync_app/serializer.py::build_routing_key` from the resolved owner and name, and that call happens inside `sync_app/service.py::sync_item` after `_resolve_owner` returns. This directly contradicts the support note claim that `routing_key` is computed before the CLI-applied owner is in effect.

The explicit and default CLI samples both emit `pm-oncall:launch-checklist`, but that is because the configured default owner in `config/defaults.json` is also `pm-oncall`, not because the path bypasses `--owner`. The distinction remains visible in `owner_source`, which is asserted as `"explicit"` and `"default"` in `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`.

## Test And Runtime Evidence

`tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path emits `owner="pm-oncall"`, `owner_source="explicit"`, and `routing_key="pm-oncall:launch-checklist"` after `--owner` is provided.

`tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` confirms the fallback path emits `owner_source="default"` and a routing key built from the default owner loaded by `sync_app/service.py::_load_defaults`.

Observed CLI output from `python -m sync_app.cli --name 'Launch Checklist' --status pending --owner pm-oncall` matched the tested explicit path, and the no-owner invocation matched the tested default path except for the same owner string coming from `config/defaults.json`.

## Conclusion

The evidence supports a docs correction, not a storage-layer behavior change. The support note should be corrected to say that storage returns only the base record, while `owner_source` and `routing_key` are service/serializer outputs from `sync_app/service.py::_resolve_owner`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.
