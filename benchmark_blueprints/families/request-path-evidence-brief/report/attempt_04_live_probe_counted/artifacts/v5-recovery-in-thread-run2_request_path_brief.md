# Request Path Brief

## Verdict

The support note is **not correct**.

## Live Path

The CLI accepts `--owner` in `sync_app/cli.py::main` and passes that value directly into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the call to `sync_app/service.py::_resolve_owner` decides both the effective `owner` value and the `owner_source` label before any storage or payload emission happens.

After that resolution step, `sync_app/service.py::sync_item` calls `sync_app/store.py::make_record`, which only creates a base record with `name`, `status`, and `owner`. The store helper does not derive `owner_source` or `routing_key`.

`routing_key` is computed later in the same service function by `sync_app/serializer.py::build_routing_key`, using the already-resolved `effective_owner`. The final payload is emitted by `sync_app/serializer.py::serialize_payload`, which copies the stored record and adds the already-derived `owner_source` and `routing_key`.

## Support Note Assessment

- The claim that `owner_source` "looks like it comes from storage" is contradicted by `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, and `sync_app/store.py::make_record`.
- The claim that `routing_key` is probably computed before the CLI applies `--owner` is contradicted by the actual call order from `sync_app/cli.py::main` to `sync_app/service.py::sync_item` to `sync_app/serializer.py::build_routing_key`.
- The current evidence points to a docs/support-note correction, not a storage-layer behavior fix.

## Test Observations

`tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all match the live service-first derivation path: explicit `--owner` yields `owner_source="explicit"` and a routing key built from that owner, while omission yields the default owner with `owner_source="default"`.
