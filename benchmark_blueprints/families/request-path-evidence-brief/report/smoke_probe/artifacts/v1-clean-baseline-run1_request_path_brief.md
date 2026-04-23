# Request Path Evidence Brief

Variant: `v1-clean-baseline`

## Conclusion

The live request path is `sync_app/cli.py::main` -> `sync_app/service.py::sync_item`. Inside `sync_item`, the owner is resolved first by `sync_app/service.py::_resolve_owner`, the base record is built by `sync_app/store.py::make_record`, the routing key is derived from the resolved owner plus `name` by `sync_app/serializer.py::build_routing_key`, and the final emitted payload is assembled by `sync_app/serializer.py::serialize_payload`.

`--owner` enters through `sync_app/cli.py::main`, which passes `args.owner` into `sync_app/service.py::sync_item`. `owner_source` is not read from storage; it is returned by `sync_app/service.py::_resolve_owner` as `"explicit"` when a non-empty owner flag is supplied and `"default"` when the flag is absent. `routing_key` is not computed before CLI owner application; it is computed later inside `sync_app/service.py::sync_item` from `effective_owner` after owner resolution by `sync_app/serializer.py::build_routing_key`.

## Live Path

1. CLI parsing accepts `--owner` in `sync_app/cli.py::main`.
2. The CLI passes `args.owner` to the service in `sync_app/cli.py::main`.
3. The service resolves `(effective_owner, owner_source)` in `sync_app/service.py::_resolve_owner`.
4. The service persists only the base record fields via `sync_app/store.py::make_record`.
5. The service derives `routing_key` from `effective_owner` and `name` via `sync_app/serializer.py::build_routing_key`.
6. The service emits the final payload by copying the record and attaching `owner_source` and `routing_key` in `sync_app/serializer.py::serialize_payload`.
7. The CLI returns JSON only after the service has already produced the full payload in `sync_app/cli.py::main`.

## Field Derivations

### `owner_source`

`owner_source` is derived only in `sync_app/service.py::_resolve_owner`. The function returns `"explicit"` when `owner` is present and non-blank, otherwise it loads the default owner from `config/defaults.json` through `sync_app/service.py::_load_defaults` and returns `"default"`. The value is then attached during emission in `sync_app/serializer.py::serialize_payload`.

### `routing_key`

`routing_key` is derived in `sync_app/service.py::sync_item` by calling `sync_app/serializer.py::build_routing_key(effective_owner, name)`. The serializer slugifies both inputs inside `sync_app/serializer.py::_slugify` and formats the key as `owner_slug:name_slug` in `sync_app/serializer.py::build_routing_key`.

### Emission

The emitted payload is produced in `sync_app/serializer.py::serialize_payload`, which starts from the record returned by `sync_app/store.py::make_record` and then adds `owner_source` and `routing_key`. JSON encoding happens later in `sync_app/cli.py::main`, so emission of these fields occurs before CLI output formatting.

## Test Observations

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` shows the explicit-owner case emits `owner_source="explicit"` and `routing_key="pm-oncall:launch-checklist"` from `sync_app/service.py::sync_item`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` shows the no-flag case falls back to `config/defaults.json` and emits `owner_source="default"` and a routing key prefixed by the default owner, confirming `sync_app/service.py::_resolve_owner`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` shows the CLI path preserves the same service-derived values through `sync_app/cli.py::main`.
- `tests/test_docs.py::test_docs_reference_owner_path_surfaces` confirms the current docs mention the path surfaces, but it does not replace code evidence for derivation order.

## Rejected Decoys

- `ops/support_note.md` is a decoy. Its claims are phrased as speculation and conflict with the live code path: `owner_source` does not come from storage, and `routing_key` is not computed before owner resolution.
- `sync_app/store.py::legacy_make_record_with_owner_source` is a decoy because the live path imports and calls `sync_app/store.py::make_record`, not the legacy helper.
- `sync_app/store.py::legacy_build_routing_key` is a decoy because the live path imports and calls `sync_app/serializer.py::build_routing_key`.
- `sync_app/serializer.py::draft_owner_source_from_record` is a decoy because no live caller reaches it, and the emitted payload source is `sync_app/service.py::_resolve_owner` plus `sync_app/serializer.py::serialize_payload`.
