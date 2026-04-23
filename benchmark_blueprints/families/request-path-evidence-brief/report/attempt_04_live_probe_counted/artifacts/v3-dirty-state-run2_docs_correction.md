# Docs Correction

## Recommended correction

The repo evidence supports a docs-only correction, not a storage-layer fix.

- Replace the storage-oriented suspicion from `ops/support_note.md` with the service/serializer path documented by `docs/data_flow.md`.
- Keep the owner fallback statement in `docs/cli.md` because it matches `sync_app/service.py::_resolve_owner`.

## Corrected wording

Suggested replacement for the support note:

> `--owner` is parsed in `sync_app/cli.py::main` and forwarded into `sync_app/service.py::sync_item`. The service resolves `effective_owner` and `owner_source` in `sync_app/service.py::_resolve_owner`, stores only the base record through `sync_app/store.py::make_record`, computes `routing_key` from the resolved owner in `sync_app/serializer.py::build_routing_key`, and emits `owner_source` plus `routing_key` in `sync_app/serializer.py::serialize_payload`.

## Why this is the smallest accurate fix

- `sync_app/store.py::make_record` cannot be the source of `owner_source` because it returns only `name`, `status`, and `owner`.
- `sync_app/serializer.py::serialize_payload` is the first live point where `owner_source` and `routing_key` are attached to the payload.
- `sync_app/cli.py::main` passes `args.owner` directly to `sync_app/service.py::sync_item`, so the repo does not support the claim that `routing_key` is computed before the CLI applies `--owner`.
