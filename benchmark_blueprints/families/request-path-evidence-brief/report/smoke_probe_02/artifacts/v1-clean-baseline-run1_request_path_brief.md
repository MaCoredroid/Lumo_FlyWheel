# Request Path Brief

Variant: `v1-clean-baseline`

Support note verdict: The support note in `ops/support_note.md::Escalation Summary` is not correct as written. `--owner` is parsed in `sync_app/cli.py::main` and passed directly into `sync_app/service.py::sync_item`. The service resolves both the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner` before it builds the stored record or derives `routing_key`. The stored record from `sync_app/store.py::make_record` contains only `name`, `status`, and `owner`; `owner_source` and `routing_key` are added later by `sync_app/serializer.py::serialize_payload`.

Live path:

1. `sync_app/cli.py::main` accepts `--owner` and calls `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, which returns `(effective_owner, owner_source)` from either the stripped CLI value or `config/defaults.json` via `sync_app/service.py::_load_defaults`.
3. `sync_app/service.py::sync_item` passes `effective_owner` into `sync_app/store.py::make_record`, so storage receives only the resolved `owner`.
4. `sync_app/service.py::sync_item` then calls `sync_app/serializer.py::build_routing_key` with the same `effective_owner` and `name`, so `routing_key` is derived after owner resolution, not before CLI owner handling.
5. `sync_app/service.py::sync_item` finally calls `sync_app/serializer.py::serialize_payload`, which copies the record and appends `owner_source` and `routing_key` to the emitted payload.

What repo tests prove:

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` asserts that explicit `--owner` input yields `owner="pm-oncall"`, `owner_source="explicit"`, and `routing_key="pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` asserts that omitting `--owner` yields the default owner plus `owner_source="default"` and the default-owner routing key.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI flag reaches the emitted JSON payload intact through `sync_app/cli.py::main`.

Conclusion:

The support note's two causal claims are contradicted by the live code path. `owner_source` is not sourced from storage, and `routing_key` is not computed before `--owner` is applied. The correction belongs in the note/docs interpretation, not in the storage layer. `docs/data_flow.md::Data Flow Note` already matches the implementation order in `sync_app/service.py::sync_item`.
