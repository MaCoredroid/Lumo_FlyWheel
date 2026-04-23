# Request Path Evidence Brief

## Verdict

The support note is **not correct**.

- `owner_source` does not come from storage. It is derived in `sync_app/service.py::_resolve_owner`, then passed into `sync_app/serializer.py::serialize_payload`. The live store helper `sync_app/store.py::make_record` persists only `name`, `status`, and `owner`.
- `routing_key` is not computed before the CLI applies `--owner`. The CLI passes `args.owner` into `sync_app/service.py::sync_item` from `sync_app/cli.py::main`, and `sync_app/service.py::sync_item` computes `routing_key` from the already-resolved `effective_owner` via `sync_app/serializer.py::build_routing_key`.

## Live Path

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item` with `owner=args.owner`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` to produce `(effective_owner, owner_source)`.
3. `sync_app/service.py::sync_item` calls `sync_app/store.py::make_record`, which stores only the base record fields and does not write `owner_source` or `routing_key`.
4. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::build_routing_key` using `effective_owner` and `name`.
5. `sync_app/service.py::sync_item` calls `sync_app/serializer.py::serialize_payload`, which appends `owner_source` and `routing_key` to a copy of the stored record.
6. `sync_app/cli.py::main` JSON-encodes the returned payload for emission.

## Field Conclusions

- `owner_source`: derived in service logic, not loaded back from storage. Evidence: `sync_app/service.py::_resolve_owner`, `sync_app/store.py::make_record`, `sync_app/serializer.py::serialize_payload`.
- `routing_key`: derived in service/serializer flow after owner resolution, using the resolved owner. Evidence: `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, `sync_app/serializer.py::build_routing_key`.
- Emission: the final payload is assembled before CLI JSON output. Evidence: `sync_app/service.py::sync_item`, `sync_app/serializer.py::serialize_payload`, `sync_app/cli.py::main`.

## Test Observations

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` proves an explicit owner yields `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` proves the default-owner fallback yields `owner_source == "default"` and a routing key built from the default owner.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` proves the CLI `--owner` flag reaches the emitted payload unchanged.

## Scope Of Fix

This is a docs/explanation correction, not a storage-layer behavior fix. The live code path already matches the corrected explanation in `docs/data_flow.md`.
