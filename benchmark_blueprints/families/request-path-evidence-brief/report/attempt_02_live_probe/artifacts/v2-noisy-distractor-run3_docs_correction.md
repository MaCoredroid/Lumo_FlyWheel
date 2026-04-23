# Docs Correction

The support note in `ops/support_note.md` should be corrected as a docs issue, not treated as evidence of a storage-layer bug.

## Corrected Statement

`--owner` is parsed in `sync_app/cli.py::main` and passed directly into `sync_app/service.py::sync_item`. `sync_app/service.py::_resolve_owner` decides both the effective owner and `owner_source`. `sync_app/store.py::make_record` stores only the base record fields `name`, `status`, and `owner`. `sync_app/serializer.py::build_routing_key` derives `routing_key` from the resolved owner and item name, and `sync_app/serializer.py::serialize_payload` appends both `owner_source` and `routing_key` to the emitted payload.

## Why The Original Note Drifted

The stored record already contains `owner`, which can make the later emitted payload look as if all ownership metadata came back from storage. Repo-local code contradicts that shortcut:
- `sync_app/store.py::make_record` does not write `owner_source`.
- `sync_app/service.py::sync_item` computes `routing_key` after owner resolution, not before the CLI flag is applied.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirm the explicit-owner path.

## Suggested Replacement For The Support Note

Replace the two speculative bullets with:

`owner_source` is derived in `sync_app/service.py::_resolve_owner`, not recovered from storage. `routing_key` is computed in `sync_app/service.py::sync_item` by calling `sync_app/serializer.py::build_routing_key` after the effective owner has been resolved from `--owner` or defaults. Any follow-up should be documentation-oriented unless the live call path changes.
