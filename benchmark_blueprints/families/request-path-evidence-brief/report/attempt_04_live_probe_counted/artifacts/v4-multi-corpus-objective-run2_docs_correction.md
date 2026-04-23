# Docs Correction

## Correction To Apply

Replace the support-note explanation with this repo-backed summary:

`--owner` enters through `sync_app/cli.py::main` and is passed directly to `sync_app/service.py::sync_item`. `sync_app/service.py::_resolve_owner` decides both the effective owner value and the `owner_source` label. `sync_app/store.py::make_record` persists only the base record fields (`name`, `status`, `owner`). `sync_app/serializer.py::build_routing_key` derives `routing_key` from the resolved owner and item name, and `sync_app/serializer.py::serialize_payload` appends `owner_source` and `routing_key` to the emitted payload.

## What Should Be Removed

- Remove the claim that `owner_source` "comes from storage". `sync_app/store.py::make_record` does not persist that field, while `sync_app/serializer.py::serialize_payload` appends it during emission.
- Remove the claim that `routing_key` is computed before `--owner` is applied. `sync_app/cli.py::main` passes the parsed flag to `sync_app/service.py::sync_item`, and `sync_app/service.py::sync_item` computes `routing_key` only after `sync_app/service.py::_resolve_owner` returns the effective owner.
- Do not treat the legacy helpers in `sync_app/store.py::legacy_make_record_with_owner_source` and `sync_app/store.py::legacy_build_routing_key` as the live path.

## Evidence Checks

- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` proves the explicit-owner CLI path emits `owner_source="explicit"` and `routing_key="pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` proves the default-owner path emits `owner_source="default"` while still using the resolved owner for `routing_key`.
- `sync_app/serializer.py::draft_owner_source_from_record` and the plan note in `release_context/future_serializer_split.md` are not live derivation evidence for the exported payload path.
