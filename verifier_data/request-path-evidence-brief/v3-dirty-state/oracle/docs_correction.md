`ops/support_note.md::escalation_summary` is incorrect.

The live code resolves the effective owner in `sync_app/service.py::_resolve_owner`, stores only the base record via `sync_app/store.py::make_record`, derives `routing_key` in `sync_app/serializer.py::build_routing_key`, and emits both fields in `sync_app/serializer.py::serialize_payload`.
