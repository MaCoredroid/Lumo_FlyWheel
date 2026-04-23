# Request Path Brief

Support note verdict: the support note is incorrect.

The live path starts at `sync_app/cli.py::main`, where `--owner` is parsed and forwarded into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the service calls `sync_app/service.py::_resolve_owner` before record creation, so `owner_source` is decided in the service layer, not recovered from storage. When `--owner` is present and non-blank, `sync_app/service.py::_resolve_owner` returns the stripped CLI value plus `"explicit"`; otherwise it loads the default owner through `sync_app/service.py::_load_defaults` and returns `"default"`.

After owner resolution, `sync_app/service.py::sync_item` builds the base record through `sync_app/store.py::make_record`, computes `routing_key` through `sync_app/serializer.py::build_routing_key`, and only then emits the final payload through `sync_app/serializer.py::serialize_payload`. That means the support note’s second claim is also wrong: `routing_key` is computed after the CLI-supplied owner has already been applied by the service path, not before. The serializer only appends fields to the emitted payload; it does not infer `owner_source` from the stored record.

Repo-local tests match that reading. `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` both expect `"owner_source": "explicit"` and `"routing_key": "pm-oncall:launch-checklist"` when `--owner pm-oncall` is supplied. `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` expects `"owner_source": "default"` and a routing key derived from the config default when the flag is absent.

Conclusion: the corrective action is docs-only. The live behavior in `sync_app/cli.py::main`, `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner`, `sync_app/store.py::make_record`, `sync_app/serializer.py::build_routing_key`, and `sync_app/serializer.py::serialize_payload` is internally consistent. The stale support explanation should be replaced rather than moving logic into storage.
