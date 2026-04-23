# Docs Correction

## Corrected Statement

The support note should be corrected to say that the live implementation resolves the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner`, stores only the base record in `sync_app/store.py::make_record`, then derives `routing_key` in `sync_app/serializer.py::build_routing_key` and emits the final payload in `sync_app/serializer.py::serialize_payload`.

## What To Remove From The Explanation

- Do not say storage decides `owner_source`; the live path shows storage only receives an already-resolved `owner` via `sync_app/service.py::sync_item` and `sync_app/store.py::make_record`.
- Do not say `routing_key` is computed before CLI owner precedence is applied; `sync_app/cli.py::main` forwards `--owner` into `sync_app/service.py::sync_item`, and only after that does `sync_app/serializer.py::build_routing_key` run.

## Short Replacement Paragraph

Suggested replacement: the current code path is CLI -> service owner resolution -> store base record -> serializer routing derivation -> serializer payload emission. `owner_source` is decided in the service layer, `routing_key` is built from the resolved owner, and the store layer is not the source of truth for either derived field. This reading is reinforced by `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`.
