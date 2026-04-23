# Docs Correction

The correction belongs in the support note, not in the storage layer.

Suggested correction for the statement in `ops/support_note.md`:

- Replace the claim that `owner_source` "looks like it comes from storage" with: `owner_source` is decided in `sync_app/service.py::_resolve_owner` and only attached during payload emission in `sync_app/serializer.py::serialize_payload`.
- Replace the claim that `routing_key` is "probably computed before the CLI applies --owner" with: `sync_app/cli.py::main` forwards `--owner` into `sync_app/service.py::sync_item`, and `routing_key` is computed there via `sync_app/serializer.py::build_routing_key` after owner resolution.
- Replace the proposed fix scope with: no storage-layer code fix is indicated by repo-local evidence; the issue is a misleading support note. The current repo docs in `docs/data_flow.md::Data Flow Note` and `docs/cli.md::project board sync CLI` already match the implemented flow.

Why this correction is supported:

- `sync_app/store.py::make_record` only persists `name`, `status`, and `owner`, so it is not the source of `owner_source`.
- `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` before it calls `sync_app/serializer.py::build_routing_key`.
- `sync_app/serializer.py::serialize_payload` is the live emission point for both `owner_source` and `routing_key`.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` validate the implemented ordering.
