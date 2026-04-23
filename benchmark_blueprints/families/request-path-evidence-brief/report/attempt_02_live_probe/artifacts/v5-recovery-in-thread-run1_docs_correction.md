# Docs Correction

The support note in `ops/support_note.md` should be corrected to say that the live implementation resolves owner precedence in `sync_app/service.py::_resolve_owner`, persists only the base record through `sync_app/store.py::make_record`, computes `routing_key` from the resolved owner in `sync_app/serializer.py::build_routing_key`, and emits `owner_source` plus `routing_key` in `sync_app/serializer.py::serialize_payload`.

The docs fix belongs in documentation, not in repo behavior. Repo-local code already matches the documented flow in `docs/cli.md` and `docs/data_flow.md`, and the behavior is covered by `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`.

Suggested replacement for the support note:

`owner_source` is decided in `sync_app/service.py::_resolve_owner`, not in storage. `routing_key` is built in `sync_app/serializer.py::build_routing_key` from the resolved owner after CLI owner precedence has already been applied by `sync_app/service.py::sync_item`. `sync_app/store.py::make_record` stores the base record only and does not decide either derived field.
