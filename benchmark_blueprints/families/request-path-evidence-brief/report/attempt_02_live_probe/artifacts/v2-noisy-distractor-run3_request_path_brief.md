# Request Path Evidence Brief

Variant: `v2-noisy-distractor`

## Verdict

The support note is **not correct**. Repo-local code shows that `--owner` is parsed in `sync_app/cli.py::main`, passed directly into `sync_app/service.py::sync_item`, resolved into both `effective_owner` and `owner_source` by `sync_app/service.py::_resolve_owner`, and only then used to create the stored record, compute `routing_key`, and emit the payload through `sync_app/serializer.py::serialize_payload`.

## Live Request Path

The live path is:
`sync_app/cli.py::main` -> `sync_app/service.py::sync_item` -> `sync_app/service.py::_resolve_owner` -> `sync_app/store.py::make_record` and `sync_app/serializer.py::build_routing_key` -> `sync_app/serializer.py::serialize_payload`.

Evidence:
- `sync_app/cli.py::main` defines `--owner` and passes `args.owner` into `sync_app/service.py::sync_item`.
- `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when the flag is present and otherwise returns the default owner with `"default"`.
- `sync_app/service.py::sync_item` calls `sync_app/store.py::make_record` with the already-resolved owner, then computes `routing_key` from that same resolved owner via `sync_app/serializer.py::build_routing_key`, then appends both derived fields in `sync_app/serializer.py::serialize_payload`.
- `sync_app/store.py::make_record` stores only `name`, `status`, and `owner`; it does not derive or persist `owner_source` or `routing_key`.

## Support Note Check

Claim 1 from `ops/support_note.md` is not supported. The presence of `owner` in the record does not imply `owner_source` came from storage, because `sync_app/store.py::make_record` does not write `owner_source`, while `sync_app/serializer.py::serialize_payload` adds it at emission time from the value produced earlier by `sync_app/service.py::_resolve_owner`.

Claim 2 from `ops/support_note.md` is also not supported. `sync_app/cli.py::main` passes the CLI flag into `sync_app/service.py::sync_item`, and `sync_app/service.py::sync_item` computes `routing_key` only after resolving that owner. The explicit-owner runtime output matched this path and emitted `routing_key = "pm-oncall:launch-checklist"` with `owner_source = "explicit"`.

## Tests And Runtime Observations

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` asserts that `sync_app/service.py::sync_item` returns `owner_source = "explicit"` and `routing_key = "pm-oncall:launch-checklist"` when `owner="pm-oncall"` is supplied.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` asserts that omitted `--owner` falls back to `config/defaults.json` through `sync_app/service.py::_load_defaults` and marks `owner_source = "default"`.
- Running `python -m sync_app.cli --name 'Launch Checklist' --status pending --owner pm-oncall` emitted JSON with `owner = "pm-oncall"`, `owner_source = "explicit"`, and `routing_key = "pm-oncall:launch-checklist"`, consistent with `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, and `sync_app/serializer.py::serialize_payload`.

## Conclusion

The repo-local evidence points to a docs correction, not a storage-layer fix. `docs/data_flow.md` is directionally correct that historical notes over-compressed the flow, and the concrete correction is that `owner_source` and `routing_key` are derived in service/serializer code after owner resolution, not recovered from storage and not computed before the CLI-applied owner reaches the service.
