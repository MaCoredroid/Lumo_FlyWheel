# Docs Correction

The support note in `ops/support_note.md::Escalation Summary` should be corrected. Based on the live path in `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`, the storage layer is not the source of `owner_source`, and `routing_key` is not computed before `--owner` is applied.

## Corrected Statement

- `owner_source` is derived in `sync_app/service.py::_resolve_owner`, before serialization, from either the explicit `--owner` value or the default owner loaded by `sync_app/service.py::_load_defaults`.
- The base stored record from `sync_app/store.py::make_record` contains `name`, `status`, and `owner` only.
- `routing_key` is derived in `sync_app/serializer.py::build_routing_key` after `sync_app/cli.py::main` has passed `args.owner` into `sync_app/service.py::sync_item`.
- The final payload is emitted in `sync_app/serializer.py::serialize_payload`, which adds `owner_source` and `routing_key` onto the base record.

## Fix Scope

- No storage-layer behavior change is indicated by repo-local evidence.
- The needed correction is documentation/support-note wording only.
- `docs/data_flow.md::Data Flow Note` already describes the high-level order correctly; the misleading artifact is the speculative wording in `ops/support_note.md::Escalation Summary`.
