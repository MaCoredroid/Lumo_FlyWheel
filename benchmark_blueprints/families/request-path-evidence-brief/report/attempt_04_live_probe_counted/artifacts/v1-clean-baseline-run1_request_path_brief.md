# Request Path Brief

## Verdict

The support note is not correct. `owner_source` does not come from storage, and `routing_key` is not computed before the CLI-applied `--owner` reaches the service. The live path is `sync_app/cli.py::main` -> `sync_app/service.py::sync_item` -> `sync_app/service.py::_resolve_owner` / `sync_app/store.py::make_record` / `sync_app/serializer.py::build_routing_key` / `sync_app/serializer.py::serialize_payload`.

## What The Repo Shows

`sync_app/cli.py::main` parses `--owner` and passes `args.owner` straight into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner` produces both the effective owner value and the `owner_source` label. The same resolved owner is then used by `sync_app/store.py::make_record` to build the base record and by `sync_app/serializer.py::build_routing_key` to derive `routing_key`. Finally, `sync_app/serializer.py::serialize_payload` appends `owner_source` and `routing_key` to the exported payload.

`sync_app/store.py::make_record` only returns `name`, `status`, and `owner`. There is no live-path storage write that sets `owner_source` or `routing_key`. The repo does contain `sync_app/store.py::legacy_make_record_with_owner_source`, `sync_app/store.py::legacy_build_routing_key`, and `sync_app/serializer.py::draft_owner_source_from_record`, but repo-wide search shows they are not called by the active flow.

## Outcome

The current evidence supports a docs correction, not a storage-layer fix. `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all confirm that the emitted payload carries service-derived `owner_source` and a `routing_key` built from the resolved owner before serialization, not from storage after the fact.
