# Request Path Evidence Brief

## Verdict

The support note is not correct for the live code path. The exported payload does not derive `owner_source` from storage, and `routing_key` is not computed before CLI owner precedence is applied. The live path is `sync_app/cli.py::main` -> `sync_app/service.py::sync_item`, where `sync_app/service.py::_resolve_owner` decides both the effective owner and `owner_source`, `sync_app/store.py::make_record` persists only the base record fields, `sync_app/serializer.py::build_routing_key` computes `routing_key` from the already-resolved owner, and `sync_app/serializer.py::serialize_payload` emits both derived fields.

## Live Request Path

`sync_app/cli.py::main` accepts `--owner` and passes `args.owner` into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when the flag is present and otherwise returns the default owner from config with `"default"`. That means the owner choice and `owner_source` are both fixed in service code before any record is created.

After resolution, `sync_app/service.py::sync_item` calls `sync_app/store.py::make_record` with only `name`, `status`, and the resolved `owner`. The live storage helper does not accept or derive `owner_source` or `routing_key`; it returns only `{"name", "status", "owner"}`. Next, the same service function calls `sync_app/serializer.py::build_routing_key` with the resolved owner and name, so `routing_key` is computed after owner precedence has already been applied. Finally, `sync_app/serializer.py::serialize_payload` copies the stored record and appends `owner_source` and `routing_key`.

## Why The Support Note Looked Plausible

The support note's first claim is a shortcut based on the stored record already containing `owner`, but `docs/data_flow.md` explicitly warns that this does not prove where `owner_source` or `routing_key` are decided, and the live code confirms that warning through `sync_app/service.py::_resolve_owner`, `sync_app/store.py::make_record`, and `sync_app/serializer.py::serialize_payload`. The second claim is contradicted directly by `sync_app/service.py::sync_item`, which computes `routing_key` only after it receives `effective_owner` from `sync_app/service.py::_resolve_owner`.

## Test Observations

`tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` shows the explicit-owner path returns `owner_source="explicit"` and `routing_key="pm-oncall:launch-checklist"`, matching service-first derivation. `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` shows the same service path falls back to `config/defaults.json` and emits `owner_source="default"` plus a routing key built from that default owner. `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI flag path flows through to the same payload shape.

## Support-Note Answer

The fix does not belong in the live storage layer based on current repo evidence. The code path already resolves owner precedence and computes both derived fields in service/serializer code, so the correction belongs in docs and support guidance, not in `sync_app/store.py::make_record`.
