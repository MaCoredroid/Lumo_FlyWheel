# Request Path Evidence Brief

Variant: `v4-multi-corpus-objective`

Support note verdict: the support note is not correct.

The live path starts at `sync_app/cli.py::main`, which parses `--owner` and passes `args.owner` into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the service calls `sync_app/service.py::_resolve_owner` first, so `owner_source` is decided in service code before serialization. The same resolved owner is then passed into `sync_app/store.py::make_record` and into `sync_app/serializer.py::build_routing_key`.

`owner_source` does not come from storage in the live path. `sync_app/store.py::make_record` returns only `name`, `status`, and `owner`, while `sync_app/serializer.py::serialize_payload` simply copies the stored record and adds the already-derived `owner_source` and `routing_key`. That matches the caution in `docs/data_flow.md` about historical notes over-compressing the flow and is reinforced by `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`.

`routing_key` is also not computed before the CLI applies `--owner`. The CLI forwards the parsed owner to `sync_app/service.py::sync_item`, and only then does `sync_app/service.py::sync_item` call `sync_app/serializer.py::build_routing_key` with the effective owner returned by `sync_app/service.py::_resolve_owner`. The direct CLI run through `sync_app/cli.py::main` emits `owner_source="explicit"` and `routing_key="pm-oncall:launch-checklist"` when `--owner pm-oncall` is present, which matches `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`.

The repo-local evidence points to a documentation/support-note correction, not a storage-layer fix. `docs/cli.md` and `docs/data_flow.md` already describe the live behavior consistently; the stale claims are in `ops/support_note.md` and the unfinished draft in `ops/previous_session_path.md`.
