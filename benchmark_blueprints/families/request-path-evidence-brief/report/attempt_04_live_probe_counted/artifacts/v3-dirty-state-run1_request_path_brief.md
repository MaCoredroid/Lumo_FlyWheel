# Request Path Evidence Brief

Variant: `v3-dirty-state`

## Verdict

The support note is **not correct**.

- The claim that `owner_source` in the emitted payload comes from storage is not supported by the live path. `sync_app/store.py::make_record` stores only `name`, `status`, and `owner`, while `sync_app/service.py::sync_item` gets `owner_source` from `sync_app/service.py::_resolve_owner` and passes it into `sync_app/serializer.py::serialize_payload`.
- The claim that `routing_key` is computed before the CLI applies `--owner` is also not supported by the live path. `sync_app/cli.py::main` passes `args.owner` into `sync_app/service.py::sync_item`, and `sync_app/service.py::sync_item` computes `routing_key` only after resolving the effective owner, via `sync_app/serializer.py::build_routing_key`.

## Live Path

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item` with `owner=args.owner`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, which returns `(owner.strip(), "explicit")` when `--owner` is present and otherwise returns `(defaults["owner"], "default")`.
3. `sync_app/service.py::sync_item` persists the base record through `sync_app/store.py::make_record`, which includes `owner` but does not include `owner_source` or `routing_key`.
4. `sync_app/service.py::sync_item` computes `routing_key` from the resolved owner and name via `sync_app/serializer.py::build_routing_key`.
5. `sync_app/service.py::sync_item` emits the final payload through `sync_app/serializer.py::serialize_payload`, which copies the stored record and then appends `owner_source` and `routing_key`.
6. `sync_app/cli.py::main` JSON-encodes that payload for emission.

## Test Evidence

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` proves the explicit-owner path yields `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` proves the default-owner path yields `owner_source == "default"` and a routing key derived from `config/defaults.json`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` proves the CLI flag reaches the emitted payload unchanged through the live path.

## Conclusion

The fix belongs in docs, not in repo behavior. The live implementation already resolves `owner_source` and `routing_key` in the service/serializer path, and the storage-layer explanation in `ops/support_note.md` should be corrected to match `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.
