# Docs Correction

## Recommended Correction

Replace the old support-note explanation with:

> `owner_source` is derived by service logic in `sync_app/service.py::_resolve_owner`, not read back from storage. The store helper in `sync_app/store.py::make_record` only returns `name`, `status`, and `owner`. `routing_key` is computed in the same live request path after owner resolution, because `sync_app/cli.py::main` passes `--owner` into `sync_app/service.py::sync_item`, which then calls `sync_app/serializer.py::build_routing_key` using the resolved owner.

## Why This Correction Is Needed

- The old support note attributes `owner_source` to storage, but the live store code in `sync_app/store.py::make_record` has no `owner_source` field.
- The old support note suggests `routing_key` may be computed before `--owner` is applied, but `sync_app/cli.py::main` forwards `args.owner` directly into `sync_app/service.py::sync_item`, and `sync_app/service.py::sync_item` computes `routing_key` only after `_resolve_owner` returns `effective_owner`.
- The existing user-facing explanation in `docs/data_flow.md` already matches the code path and should remain the reference description.

## No Behavior Change Required

No code change is indicated by this trace. The correction belongs in explanatory notes only, because the live behavior already aligns with `docs/data_flow.md` and the passing sync-path tests in `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`.
