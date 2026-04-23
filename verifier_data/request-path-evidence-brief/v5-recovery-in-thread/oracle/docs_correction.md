`incident_context/rollback_note.md::incident_follow_up` shows the earlier correction was rolled back.

The live code resolves the effective owner in `sync_app/service.py::_resolve_owner`, stores only the base record via `sync_app/store.py::make_record`, derives `routing_key` in `sync_app/serializer.py::build_routing_key`, and emits both fields in `sync_app/serializer.py::serialize_payload`. The replacement correction should explicitly say the old store-layer explanation was stale, not reassert it.
