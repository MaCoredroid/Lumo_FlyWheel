# Docs Correction

The support note in `ops/support_note.md` should be corrected to say:

`owner_source` is derived in `sync_app/service.py::_resolve_owner`, not inferred from storage. The stored record created by `sync_app/store.py::make_record` only contains `name`, `status`, and the resolved `owner`. `routing_key` is computed in `sync_app/serializer.py::build_routing_key` from the same resolved owner inside `sync_app/service.py::sync_item`, after `sync_app/cli.py::main` has already passed through `--owner`. The fix is documentation-only; the storage layer does not need a behavior change.

Evidence for the correction:
- `sync_app/cli.py::main`
- `sync_app/service.py::_resolve_owner`
- `sync_app/service.py::sync_item`
- `sync_app/store.py::make_record`
- `sync_app/serializer.py::build_routing_key`
- `sync_app/serializer.py::serialize_payload`
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`
