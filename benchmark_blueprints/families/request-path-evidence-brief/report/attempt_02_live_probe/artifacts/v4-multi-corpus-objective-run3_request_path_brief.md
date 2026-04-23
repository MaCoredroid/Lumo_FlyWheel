# Request Path Evidence Brief

Variant: `v4-multi-corpus-objective`

## Verdict

The support note is **not correct**.

- `--owner` is accepted by `sync_app/cli.py::main` and passed directly into `sync_app/service.py::sync_item`.
- `owner_source` is decided in `sync_app/service.py::_resolve_owner`, not recovered from storage or inferred after the record comes back.
- `routing_key` is built in `sync_app/serializer.py::build_routing_key` only after `sync_app/service.py::_resolve_owner` has already chosen the effective owner.
- Storage only persists the base record via `sync_app/store.py::make_record`, which contains `name`, `status`, and `owner`, but not `owner_source` or `routing_key`.

## Live Request Path

1. `sync_app/cli.py::main` parses `--name`, `--status`, and `--owner`, then calls `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` to produce both `effective_owner` and `owner_source`.
3. `sync_app/service.py::sync_item` persists the base record through `sync_app/store.py::make_record`.
4. `sync_app/service.py::sync_item` computes `routing_key` through `sync_app/serializer.py::build_routing_key` using the resolved owner.
5. `sync_app/service.py::sync_item` emits the final payload through `sync_app/serializer.py::serialize_payload`, which copies the stored record and appends `owner_source` and `routing_key`.

## Why The Support Note Fails

- The claim that `owner_source` "looks like it comes from storage" is contradicted by `sync_app/store.py::make_record` and `sync_app/serializer.py::serialize_payload`. The store helper does not write `owner_source`; the serializer receives it as an explicit argument from `sync_app/service.py::sync_item`.
- The claim that `routing_key` is computed before the CLI applies `--owner` is contradicted by the call order in `sync_app/service.py::sync_item`: `sync_app/service.py::_resolve_owner` runs before `sync_app/serializer.py::build_routing_key`.

## Test And Runtime Evidence

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` asserts that an explicit owner yields `owner_source="explicit"` and `routing_key="pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` asserts that omitting `--owner` falls back through `sync_app/service.py::_load_defaults`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path preserves the explicit owner through to the emitted payload.
- A direct CLI run from this repo emitted `owner="pm-oncall"`, `owner_source="explicit"`, and `routing_key="pm-oncall:launch-checklist"`, matching the service/test expectations rooted in `sync_app/cli.py::main` and `sync_app/service.py::sync_item`.

## Practical Correction

The fix does not belong in the storage layer. The repo-local evidence supports a docs correction only: retire or rewrite the historical support note in `ops/support_note.md` so it matches the live path already described by the code and by the caution in `docs/data_flow.md`.
