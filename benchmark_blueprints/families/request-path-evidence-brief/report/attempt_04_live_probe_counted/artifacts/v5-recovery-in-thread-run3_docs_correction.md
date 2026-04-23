# Docs Correction

The support note in `ops/support_note.md::Escalation Summary` should be corrected as follows:

- `owner_source` is derived in `sync_app/service.py::_resolve_owner`, not in storage. The live storage function `sync_app/store.py::make_record` only returns `name`, `status`, and `owner`.
- `routing_key` is computed by `sync_app/serializer.py::build_routing_key` from the already-resolved effective owner inside `sync_app/service.py::sync_item`, so it is not built before CLI owner precedence is applied.
- The live fix is documentation scope only. Repo-local evidence does not show a behavior bug in the current implementation.

Do not use these as proof of the live path:

- `sync_app/store.py::legacy_make_record_with_owner_source`
- `sync_app/store.py::legacy_build_routing_key`
- `sync_app/serializer.py::draft_owner_source_from_record`
- `release_context/future_serializer_split.md::Planned Serializer Split`
- `incident_context/prior_docs_correction.md::Prior Docs Correction`
