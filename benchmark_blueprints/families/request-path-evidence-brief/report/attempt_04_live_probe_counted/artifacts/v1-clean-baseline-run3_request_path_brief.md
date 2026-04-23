# Request Path Brief

The support note is **not correct**.

`--owner` is parsed in `sync_app/cli.py::main` and forwarded directly into `sync_app/service.py::sync_item`. Inside `sync_item`, the service resolves the effective owner and `owner_source` first via `sync_app/service.py::_resolve_owner`, then builds the base record with `sync_app/store.py::make_record`, computes `routing_key` from the already-resolved owner via `sync_app/serializer.py::build_routing_key`, and finally emits both derived fields in `sync_app/serializer.py::serialize_payload`.

## Findings

- `owner_source` does not come from storage on the live path. The stored/base record is created by `sync_app/store.py::make_record`, and that function only returns `name`, `status`, and `owner`. The live payload gets `owner_source` later when `sync_app/serializer.py::serialize_payload` copies the record and adds the field supplied by `sync_app/service.py::_resolve_owner`.
- `routing_key` is not computed before the CLI applies `--owner`. The CLI passes `args.owner` into `sync_app/service.py::sync_item` from `sync_app/cli.py::main`; then `sync_item` calls `sync_app/service.py::_resolve_owner`; only after that does it call `sync_app/serializer.py::build_routing_key` with `effective_owner`.
- The support-note question in `ops/support_note.md::Escalation Summary` resolves to docs, not storage. The behavior in `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload` is internally consistent, and the tests in `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all match that path.

## Live Path Summary

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`.
3. `sync_app/service.py::_resolve_owner` returns `(effective_owner, owner_source)` from either the explicit flag or `config/defaults.json` through `sync_app/service.py::_load_defaults`.
4. `sync_app/service.py::sync_item` passes `effective_owner` into `sync_app/store.py::make_record`.
5. `sync_app/service.py::sync_item` passes `effective_owner` and `name` into `sync_app/serializer.py::build_routing_key`.
6. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::serialize_payload`, which emits the stored fields plus `owner_source` and `routing_key`.

## Decoys Rejected

- `sync_app/serializer.py::draft_owner_source_from_record` is not on the live path; repo search finds only its definition.
- `sync_app/store.py::legacy_make_record_with_owner_source` is not on the live path; repo search finds only its definition.
- `sync_app/store.py::legacy_build_routing_key` is not on the live path; the live service imports `sync_app/serializer.py::build_routing_key` instead.

## Test Evidence

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` expects `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"` when the owner is passed explicitly.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` expects `owner_source == "default"` and a default-owner-based routing key when `--owner` is omitted.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path preserves the same derived values after parsing `--owner`.
- Local run: `pytest -q tests/test_sync.py tests/test_docs.py` passed with `4 passed`.
