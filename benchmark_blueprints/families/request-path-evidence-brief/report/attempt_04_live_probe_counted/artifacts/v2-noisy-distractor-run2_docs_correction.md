# Docs Correction

Suggested correction for the support note in `ops/support_note.md`:

`owner_source` is not recovered from storage on the live path. `sync_app/cli.py::main` forwards `--owner` to `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner` decides both the effective owner and whether the source is `explicit` or `default`, `sync_app/store.py::make_record` builds only the base record, `sync_app/serializer.py::build_routing_key` derives `routing_key` from the resolved owner and name, and `sync_app/serializer.py::serialize_payload` adds both `owner_source` and `routing_key` to the emitted payload.

Docs implication: the issue is documentation accuracy, not a storage-layer bug. If the note is retained, it should explicitly reject the shortcut inference that a stored `owner` proves where `owner_source` or `routing_key` were derived. Repo-local evidence for that correction is `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, `sync_app/serializer.py::serialize_payload`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`.
