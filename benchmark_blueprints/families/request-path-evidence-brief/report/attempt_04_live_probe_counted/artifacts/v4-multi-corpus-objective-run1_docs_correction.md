# Docs Correction

The correction belongs in the historical support note, not in repo behavior.

Suggested correction for the note in `ops/support_note.md::<document>`:

- `owner_source` is derived in `sync_app/service.py::_resolve_owner`, not loaded from storage. `sync_app/store.py::make_record` stores only `name`, `status`, and `owner`, and `sync_app/serializer.py::serialize_payload` appends `owner_source` when emitting the payload.
- `routing_key` is built in `sync_app/serializer.py::build_routing_key` from the effective owner after `sync_app/cli.py::main` has already forwarded `--owner` into `sync_app/service.py::sync_item`.
- No storage-layer fix is supported by this repo snapshot. The live path already matches `docs/data_flow.md::<document>`: resolve owner, persist the base record, then emit the payload.

Why the old note was misleading:

- `config/defaults.json::owner` is `pm-oncall`, so the sample explicit-owner and default-owner runs can emit the same `owner` and `routing_key`.
- That overlap does not mean storage produced `owner_source`, and it does not mean `routing_key` was computed before the CLI flag was applied.

Cross-checks:

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`
