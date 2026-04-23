# Docs Correction

## Correction To Support Note

The support note in `ops/support_note.md` should be corrected. Based on the live path through `sync_app/cli.py::main`, `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`:

- `owner_source` does not come from storage on the live path.
- `routing_key` is not computed before the CLI-applied `--owner` reaches the service.
- No storage-layer fix is indicated by the current repo behavior.

## Suggested Replacement Text

Replace the current support note with wording like this:

`owner_source` is derived in `sync_app/service.py::_resolve_owner` when the service decides whether the request used an explicit `--owner` or the default from `config/defaults.json`. The storage layer on the live path only returns the base record from `sync_app/store.py::make_record`, which contains `name`, `status`, and `owner`. `routing_key` is computed later in `sync_app/serializer.py::build_routing_key` using the already-resolved effective owner, and both `owner_source` and `routing_key` are attached to the emitted payload by `sync_app/serializer.py::serialize_payload`.

## Why The Note Drifted

The current default owner in `config/defaults.json` is `pm-oncall`, and the docs example in `docs/cli.md` also uses `--owner pm-oncall`. Because of that overlap, explicit-owner and default-owner runs can emit the same `owner` and `routing_key` values while differing only in `owner_source`. That can make the stored `owner` field look more decisive than it really is, but `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` show the actual derivation boundary.
