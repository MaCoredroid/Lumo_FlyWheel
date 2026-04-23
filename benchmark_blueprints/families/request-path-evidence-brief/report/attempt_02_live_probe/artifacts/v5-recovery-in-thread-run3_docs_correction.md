# Docs Correction

Correct the support-thread explanation to say:

`--owner` enters at `sync_app/cli.py::main` and is forwarded into `sync_app/service.py::sync_item`. The service resolves the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner`, persists only the base record through `sync_app/store.py::make_record`, computes `routing_key` with the resolved owner via `sync_app/serializer.py::build_routing_key`, and emits both derived fields in `sync_app/serializer.py::serialize_payload`.

Do not describe `owner_source` as storage-derived. Do not describe `routing_key` as computed before CLI owner precedence. Those were stale theories called out by `incident_context/prior_docs_correction.md` and `incident_context/rollback_note.md`. Do not use `release_context/future_serializer_split.md` as evidence for live behavior because it explicitly describes future intent only.

If a short operator-facing correction is needed, use this wording:

The support note was incorrect. In the live code, `owner_source` is resolved in `sync_app/service.py::_resolve_owner`, `routing_key` is built later in `sync_app/serializer.py::build_routing_key` using the already-resolved owner, and storage (`sync_app/store.py::make_record`) only carries the base `owner` field.
