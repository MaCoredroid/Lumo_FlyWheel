# Docs Correction

The support note in `ops/support_note.md` should be corrected; the live docs in `docs/cli.md` and `docs/data_flow.md` already match the implemented request path.

Recommended replacement for the support note:

- `--owner` is parsed in `sync_app/cli.py::main` and passed directly into `sync_app/service.py::sync_item`.
- `owner_source` is derived in `sync_app/service.py::_resolve_owner`, not in storage. `sync_app/store.py::make_record` only stores `name`, `status`, and `owner`.
- `routing_key` is computed in `sync_app/service.py::sync_item` by calling `sync_app/serializer.py::build_routing_key` with the resolved owner after owner resolution, not before CLI owner handling.
- The payload is emitted in `sync_app/serializer.py::serialize_payload`, which adds the already-derived `owner_source` and `routing_key` onto the stored record.
- The likely action is documentation/support-note cleanup only, not a storage-layer code fix.

Why this correction is warranted:

- `docs/data_flow.md` already warns that historical notes can over-compress the fact that the stored record contains `owner` while still failing to prove where `owner_source` and `routing_key` are decided.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all match the live service-first derivation path.
