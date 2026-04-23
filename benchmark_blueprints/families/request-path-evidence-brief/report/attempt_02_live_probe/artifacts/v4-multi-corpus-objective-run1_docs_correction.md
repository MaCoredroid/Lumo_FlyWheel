# Docs Correction

## Scope

Correct `ops/support_note.md` so it matches the live request path implemented by `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.

## Replace the support note with

- `owner_source` is not read back from storage. It is derived in `sync_app/service.py::_resolve_owner` and attached during emission by `sync_app/serializer.py::serialize_payload`.
- `routing_key` is not computed before `--owner` is applied. `sync_app/cli.py::main` forwards the parsed flag into `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key` runs after owner resolution using the effective owner plus item name.
- The live storage helper `sync_app/store.py::make_record` only creates the base record with `name`, `status`, and `owner`.
- Based on the live code path, the fix belongs in docs, not in the storage layer.

## Notes on Decoys

Do not reuse `ops/previous_session_path.md`; it is an unfinished draft that says it was never merged. Do not treat `release_context/future_serializer_split.md` as live behavior; it explicitly says it is design intent only. Ignore `sync_app/store.py::legacy_make_record_with_owner_source` and `sync_app/store.py::legacy_build_routing_key` for this audit because `sync_app/service.py::sync_item` does not call them.
