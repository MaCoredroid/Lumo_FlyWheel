# Docs Correction

## Recommended Correction For The Support Note

Replace the speculative bullets in `ops/support_note.md::Escalation Summary` with a repo-backed statement such as:

> `owner_source` is not read back from storage in the live path. `sync_app/service.py::_resolve_owner` decides both the effective owner and whether the source is `explicit` or `default`, `sync_app/store.py::make_record` stores only `name`, `status`, and `owner`, `sync_app/serializer.py::build_routing_key` derives `routing_key` from the already-resolved owner plus name, and `sync_app/serializer.py::serialize_payload` appends both fields into the emitted payload.

## Why This Is The Correct Fix

- `sync_app/cli.py::main` passes `args.owner` directly into `sync_app/service.py::sync_item`, so the CLI applies `--owner` before any routing-key computation.
- `sync_app/service.py::_resolve_owner` is the only live component that produces `owner_source`.
- `sync_app/store.py::make_record` does not create either `owner_source` or `routing_key`.
- `sync_app/serializer.py::build_routing_key` runs after owner resolution, and `sync_app/serializer.py::serialize_payload` appends both derived fields to the emitted payload.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` and `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` already codify the correct behavior.

## Note About The Misleading Observation

The current default owner in `config/defaults.json` is `pm-oncall`, which is the same owner used in the explicit CLI example. Because of that, the explicit-owner and default-owner runs can emit the same `routing_key` string even though they take different code paths and produce different `owner_source` values. `docs/data_flow.md::Data Flow Note` already captures the correct high-level explanation; the correction belongs in the support note wording, not in the storage layer.
