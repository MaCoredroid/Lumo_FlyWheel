# Request Path Brief

The support note is not correct as written.

The live path starts at `sync_app/cli.py::main`, where `--owner` is parsed and forwarded directly into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the service first calls `sync_app/service.py::_resolve_owner`, which returns both the effective owner value and the `owner_source` label. Only after that resolution does the service persist the base record via `sync_app/store.py::make_record`, and that storage helper writes only `name`, `status`, and `owner`.

`owner_source` does not come from storage on the live path. The emitted payload gets `owner_source` only when `sync_app/service.py::sync_item` passes the resolved label into `sync_app/serializer.py::serialize_payload`. The unused helper `sync_app/serializer.py::draft_owner_source_from_record` can read `owner_source` from an existing record, but nothing in the live caller chain reaches it, so it is not evidence for the exported payload path.

`routing_key` is also not computed before the CLI-applied owner takes effect. After `sync_app/service.py::_resolve_owner` returns the effective owner, `sync_app/service.py::sync_item` calls `sync_app/serializer.py::build_routing_key` with that resolved owner and the item name, then passes the result to `sync_app/serializer.py::serialize_payload`. That means an explicit `--owner` value influences both `owner` and `routing_key` before emission.

The tests line up with that path. `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` both expect `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"` when `--owner pm-oncall` is provided. `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` expects `owner_source == "default"` and a routing key derived from the default owner when the flag is absent.

Repo-local evidence supports a docs-only correction to the support note, not a storage-layer fix. The current flow note in `docs/data_flow.md` already matches the code path described by `sync_app/cli.py::main`, `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload`.
