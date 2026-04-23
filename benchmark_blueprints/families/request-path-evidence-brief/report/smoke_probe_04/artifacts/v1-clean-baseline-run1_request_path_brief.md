# Request Path Brief

## Verdict

The support note is not correct based on the live code path. `owner_source` does not come from storage, and `routing_key` is not computed before the CLI-applied `--owner` reaches the service. The live flow is `sync_app/cli.py::main` -> `sync_app/service.py::sync_item`, where `sync_app/service.py::_resolve_owner` decides both the effective owner value and the `owner_source`, `sync_app/store.py::make_record` stores only the base record fields, `sync_app/serializer.py::build_routing_key` derives `routing_key` from the already-resolved owner plus name, and `sync_app/serializer.py::serialize_payload` appends `owner_source` and `routing_key` to the emitted payload.

## Evidence

`sync_app/cli.py::main` accepts `--owner` and passes it directly into `sync_app/service.py::sync_item` as `owner=args.owner`. There is no earlier routing-key computation in the CLI.

`sync_app/service.py::_resolve_owner` returns a tuple of `(effective_owner, owner_source)`. When `--owner` is present, the tuple is `(owner.strip(), "explicit")`; otherwise it falls back to the default owner from config and returns `"default"`. That means `owner_source` is derived in the service, not recovered from storage.

`sync_app/store.py::make_record` stores only `name`, `status`, and `owner`. It never writes `owner_source` or `routing_key`. The legacy helpers `sync_app/store.py::legacy_make_record_with_owner_source` and `sync_app/store.py::legacy_build_routing_key` are decoys relative to the live path because `sync_app/service.py::sync_item` does not call them.

`sync_app/service.py::sync_item` calls `sync_app/store.py::make_record` first, then calls `sync_app/serializer.py::build_routing_key` with the already-resolved `effective_owner`, and finally calls `sync_app/serializer.py::serialize_payload` with the stored record plus the separately derived `owner_source` and `routing_key`.

`tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all confirm the same behavior: explicit `--owner` yields `"owner_source": "explicit"` and a routing key derived from that explicit owner, while omission yields `"owner_source": "default"` and a routing key derived from the configured default.

## Conclusion

The evidence supports a docs-only correction, not a storage-layer change. `docs/data_flow.md` is directionally correct already because it says the service resolves the effective owner before persistence and emission, and that the presence of `owner` in storage is not enough to prove where `owner_source` or `routing_key` are decided. The misleading statement is the support note, not the runtime code path.
