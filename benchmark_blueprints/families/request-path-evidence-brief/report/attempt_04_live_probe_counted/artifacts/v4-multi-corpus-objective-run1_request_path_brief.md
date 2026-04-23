# Request Path Brief

Verdict: the support note is not correct.

The live path starts at `sync_app/cli.py::main`, which parses `--owner` and passes `args.owner` directly into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the service first calls `sync_app/service.py::_resolve_owner` to derive both the effective owner and `owner_source`, then calls `sync_app/store.py::make_record` to build the base record, then calls `sync_app/serializer.py::build_routing_key`, and finally emits the payload through `sync_app/serializer.py::serialize_payload`.

`owner_source` does not come from storage. The storage helper `sync_app/store.py::make_record` only returns `name`, `status`, and `owner`. `owner_source` is created in `sync_app/service.py::_resolve_owner` and is only attached to the outgoing payload in `sync_app/serializer.py::serialize_payload`. The unused helper `sync_app/serializer.py::draft_owner_source_from_record` does not participate in the live path.

`routing_key` is not computed before the CLI applies `--owner`. `sync_app/cli.py::main` forwards the parsed owner into `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key` is called later inside that service function using the already-resolved effective owner. The explicit-owner test `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` and the service test `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` both expect `owner_source == "explicit"` together with `routing_key == "pm-oncall:launch-checklist"`, which matches the observed CLI output.

Why the support note likely looked plausible: `config/defaults.json::owner` is also `pm-oncall`, so the explicit-owner run and the default-owner run produce the same `owner` and `routing_key` for the sample input. The difference is still visible in `owner_source`, as shown by `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` and the observed `python -m sync_app.cli --name 'Launch Checklist' --status pending` output.

The evidence supports a docs/support-note correction, not a storage-layer fix. `docs/data_flow.md::<document>` already describes the live ordering correctly: resolve owner, persist the base record, then emit the payload.
