# Request Path Brief

## Verdict

The support note is **not correct**. Repo-local code shows that `owner_source` is not derived in storage, and `routing_key` is not computed before CLI owner precedence is applied.

## Live Path

1. `sync_app/cli.py::main` parses `--owner` and forwards `args.owner` into `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` before either payload emission or routing-key construction.
3. `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when `--owner` is present, otherwise it loads the fallback owner from `sync_app/service.py::_load_defaults`, which reads `config/defaults.json::owner`, and returns `("pm-oncall", "default")`.
4. `sync_app/service.py::sync_item` persists only the base record through `sync_app/store.py::make_record`.
5. `sync_app/service.py::sync_item` then computes `routing_key` by calling `sync_app/serializer.py::build_routing_key` with the already-resolved effective owner and request name.
6. `sync_app/serializer.py::serialize_payload` emits the final payload by copying the stored record and appending `owner_source` and `routing_key`.

## Support Note Check

- The claim that storage decides `owner_source` is contradicted by `sync_app/store.py::make_record`, which returns only `name`, `status`, and `owner`. The emitted `owner_source` field is added later by `sync_app/serializer.py::serialize_payload`, using the value already chosen by `sync_app/service.py::_resolve_owner`.
- The claim that `routing_key` is computed before CLI `--owner` precedence is applied is contradicted by `sync_app/service.py::sync_item`, which calls `sync_app/service.py::_resolve_owner` first and only then calls `sync_app/serializer.py::build_routing_key`.
- The live-path evidence points to a docs/support-note correction only. There is no repo-local evidence that a storage-layer code fix is needed.

## Test Evidence

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` proves that `sync_app/service.py::sync_item` returns `owner_source="explicit"` and `routing_key="pm-oncall:launch-checklist"` when the owner is passed in directly.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` proves that the default owner comes from `config/defaults.json::owner` and that `owner_source="default"` is set in the emitted payload.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` proves that CLI `--owner` input survives through `sync_app/cli.py::main` into the emitted payload, including `owner_source` and `routing_key`.
