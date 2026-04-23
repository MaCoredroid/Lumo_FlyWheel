# Docs Correction

## Recommended Correction

The stale support narrative should be corrected to match the live implementation:

> `--owner` is parsed by `sync_app/cli.py::main` and forwarded into `sync_app/service.py::sync_item`. `sync_app/service.py::_resolve_owner` derives both the effective owner and `owner_source`. `sync_app/store.py::make_record` stores only the base record fields (`name`, `status`, `owner`). `sync_app/serializer.py::build_routing_key` derives `routing_key` from the resolved owner and item name, and `sync_app/serializer.py::serialize_payload` emits both `owner_source` and `routing_key` alongside the base record.

## What To Correct

- Replace the claim in the stale support note that `owner_source` "looks like it comes from storage" with the service-layer derivation described above. The live code path is `sync_app/service.py::_resolve_owner`, not `sync_app/store.py::make_record`.
- Replace the claim in the stale support note that `routing_key` is "probably computed before the CLI applies --owner" with the actual order: `sync_app/cli.py::main` forwards `args.owner`, then `sync_app/service.py::sync_item` resolves the effective owner, then `sync_app/serializer.py::build_routing_key` computes `routing_key`.
- Treat the previous-session draft as stale material rather than evidence. Its "store helper likely became the source of truth" claim is contradicted by `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, and `sync_app/serializer.py::serialize_payload`.

## Current Docs Status

The checked-in product docs already align with the live path, which is confirmed by `tests/test_docs.py::test_docs_reference_owner_path_surfaces` and the implementation symbols cited above:

- `docs/data_flow.md` correctly says the CLI calls `sync_item`, which resolves the effective owner, persists the base record, and then emits the payload.
- `docs/cli.md` correctly documents `--owner`, `owner_source`, `routing_key`, and the default-owner fallback from `config/defaults.json`.

No repo behavior change is supported by the evidence. The correction is to the stale support explanation only.
