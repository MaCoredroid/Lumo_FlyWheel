# Docs Correction

The repo-local implementation supports a docs-only correction: the support note in `ops/support_note.md` should be updated, and any flow summary should describe record construction rather than storage-driven derivation.

## Corrected Statement

`--owner` is parsed by `sync_app/cli.py::main` and passed into `sync_app/service.py::sync_item`. The service resolves the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner`, builds the base record with `sync_app/store.py::make_record`, derives `routing_key` from the effective owner in `sync_app/serializer.py::build_routing_key`, and emits both derived fields in `sync_app/serializer.py::serialize_payload`.

## Why The Support Note Should Change

- `owner_source` does not come from storage. The live path derives it in `sync_app/service.py::_resolve_owner` and appends it during emission in `sync_app/serializer.py::serialize_payload`.
- `routing_key` is not computed before the CLI applies `--owner`. The explicit owner reaches `sync_app/service.py::sync_item` first, and that resolved owner is the input to `sync_app/serializer.py::build_routing_key`.
- `sync_app/store.py::make_record` only builds `name`, `status`, and `owner`, so there is no storage-layer fix indicated by repo-local evidence.

## Suggested Wording

Suggested replacement for the support note:

> The payload fields `owner_source` and `routing_key` are derived in the service/serializer path, not in storage. `--owner` is applied before `routing_key` is built. This looks like a documentation clarification, not a storage-layer fix.

Suggested wording adjustment for `docs/data_flow.md`:

> The CLI entrypoint calls `sync_item`, which resolves the effective owner, builds the base record, and then emits the payload.
