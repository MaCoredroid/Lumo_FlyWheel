# Docs Correction

Suggested correction for the support note in `ops/support_note.md`:

`owner_source` is derived in `sync_app/service.py::_resolve_owner`, not in storage. `sync_app/store.py::make_record` persists only `name`, `status`, and resolved `owner`. `routing_key` is computed later in `sync_app/service.py::sync_item` by calling `sync_app/serializer.py::build_routing_key` with the already-resolved owner, after `sync_app/cli.py::main` has forwarded `--owner` into the service. Both derived fields are emitted by `sync_app/serializer.py::serialize_payload`.

Concrete correction:

- Replace the claim that `owner_source` "looks like it comes from storage" with: `owner_source` is decided in `sync_app/service.py::_resolve_owner` and attached during `sync_app/serializer.py::serialize_payload`.
- Replace the claim that `routing_key` is "probably computed before the CLI applies --owner" with: `sync_app/cli.py::main` passes `args.owner` into `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key` runs afterward from the resolved owner.
- Replace the storage-fix question with: current code already matches `docs/data_flow.md`; the needed correction is to the support note, not the storage layer.
