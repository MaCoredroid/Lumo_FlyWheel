# Docs Correction

## Correction Summary

The support note should be corrected. The live repo path shows a docs-only issue, not a storage-layer behavior bug.

## Replace The Support Explanation With This

- `sync_app/cli.py::main` parses `--owner` and passes it directly to `sync_app/service.py::sync_item`.
- `sync_app/service.py::_resolve_owner` is the place where both the effective owner and `owner_source` are decided. It returns `owner_source="explicit"` for a non-blank CLI owner and otherwise loads `config/defaults.json::owner` through `sync_app/service.py::_load_defaults` and returns `owner_source="default"`.
- `sync_app/store.py::make_record` stores only the base record fields `name`, `status`, and `owner`. It does not derive `owner_source` or `routing_key`.
- `sync_app/service.py::sync_item` computes `routing_key` only after owner resolution, by calling `sync_app/serializer.py::build_routing_key` with the resolved owner and request name.
- `sync_app/serializer.py::serialize_payload` emits the final payload by copying the stored record and appending `owner_source` and `routing_key`.

## What To Avoid Repeating

- Do not say that storage decides `owner_source`; that theory is contradicted by `sync_app/store.py::make_record` and by the emission step in `sync_app/serializer.py::serialize_payload`.
- Do not say that `routing_key` is built before CLI `--owner` precedence; `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` before `sync_app/serializer.py::build_routing_key`.
- Do not treat `sync_app/store.py::legacy_make_record_with_owner_source`, `sync_app/store.py::legacy_build_routing_key`, or `sync_app/serializer.py::draft_owner_source_from_record` as live-path evidence. Repo-local search finds no live call sites for those helpers.
- Do not cite `release_context/future_serializer_split.md` as proof of current behavior; that file labels itself as design intent only.

## Recommended Disposition

Correct the support note and keep implementation unchanged. Repo-local code and tests point to documentation cleanup, not to a storage-layer fix.
