# Docs Correction

Correction target: `ops/support_note.md::Escalation Summary`

Recommended correction text:

- `owner_source` does not come from storage. It is derived in `sync_app/service.py::_resolve_owner` and only attached to the outgoing payload in `sync_app/serializer.py::serialize_payload`.
- `routing_key` is not computed before the CLI applies `--owner`. `sync_app/cli.py::main` passes the parsed owner into `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key` runs after that owner has been resolved.
- `sync_app/store.py::make_record` is not the fix location for either field because it stores only `name`, `status`, and `owner`.
- `docs/data_flow.md::Data Flow Note` already describes the correct order: resolve owner first, persist the base record second, emit the payload last.

Net correction:

The stale interpretation is in the support note, not in the live code path. If any documentation is updated, it should clarify that the service layer owns `owner_source` resolution and `routing_key` derivation, while the storage layer only carries the resolved `owner`.
