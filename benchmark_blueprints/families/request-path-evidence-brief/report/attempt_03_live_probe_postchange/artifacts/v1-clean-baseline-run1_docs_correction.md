# Docs Correction

## Recommended Correction To The Support Note

Replace the support note with:

> `owner_source` is derived by `sync_app/service.py::_resolve_owner`, after the CLI passes `--owner` through `sync_app/cli.py::main` into `sync_app/service.py::sync_item`.
>
> Storage only produces the base record through `sync_app/store.py::make_record`; it does not derive `owner_source` or `routing_key`.
>
> `routing_key` is computed in `sync_app/serializer.py::build_routing_key` from the resolved owner and item name, and then emitted by `sync_app/serializer.py::serialize_payload`.
>
> Any fix needed here is a documentation clarification, not a storage-layer behavior change.

## Why This Correction Is Needed

The existing support note infers too much from the fact that the stored record already contains `owner`. That inference is rejected by the live code path: `sync_app/store.py::make_record` only returns `name`, `status`, and `owner`, while `sync_app/serializer.py::serialize_payload` is the point where `owner_source` and `routing_key` are attached.

The repo already hints at this in `docs/data_flow.md`, which says the stored-record shortcut is not enough to prove where `owner_source` or `routing_key` are decided. The support note should be brought in line with the actual flow implemented by `sync_app/service.py::sync_item`.
