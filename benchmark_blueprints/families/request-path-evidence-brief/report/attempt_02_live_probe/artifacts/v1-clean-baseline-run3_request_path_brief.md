# Request Path Evidence Brief

Variant: `v1-clean-baseline`

The support note is **not correct**.

The live path starts at `sync_app/cli.py::main`, where `--owner` is parsed and passed directly into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the service first calls `sync_app/service.py::_resolve_owner`, which returns both the effective owner value and `owner_source`. Only after that does the service build the stored base record with `sync_app/store.py::make_record`, compute `routing_key` with `sync_app/serializer.py::build_routing_key`, and emit the final payload with `sync_app/serializer.py::serialize_payload`.

That means `owner_source` does not come from storage. The storage-layer record built by `sync_app/store.py::make_record` contains only `name`, `status`, and `owner`; it has no logic for deriving `owner_source`. The only live emission of `owner_source` is in `sync_app/serializer.py::serialize_payload`, using the value already decided in `sync_app/service.py::_resolve_owner`. The helper `sync_app/serializer.py::draft_owner_source_from_record` exists, but nothing in the live call path uses it.

The note is also wrong about `routing_key`. `sync_app/cli.py::main` passes `args.owner` into `sync_app/service.py::sync_item`, and `sync_app/service.py::sync_item` computes `routing_key` from `effective_owner` after owner resolution by calling `sync_app/serializer.py::build_routing_key`. The tests in `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirm both the explicit-owner and default-owner cases.

Conclusion: the evidence points to a documentation/support-note misunderstanding, not a storage-layer bug. `docs/data_flow.md` already describes the flow correctly via `sync_app/service.py::sync_item`; the correction belongs in the support note wording, not in repo behavior.
