# Request Path Evidence Brief

## Verdict

The support note is **not correct**. `--owner` is parsed in `sync_app/cli.py::main`, passed into `sync_app/service.py::sync_item`, resolved into `(effective_owner, owner_source)` by `sync_app/service.py::_resolve_owner`, stored only as `owner` by `sync_app/store.py::make_record`, and then added to the emitted payload by `sync_app/serializer.py::serialize_payload`. `routing_key` is computed from the resolved owner inside `sync_app/service.py::sync_item` via `sync_app/serializer.py::build_routing_key`, after the CLI has already supplied the `owner` argument.

## Live Path

1. `sync_app/cli.py::main` defines `--owner`, parses it, and calls `sync_app/service.py::sync_item` with `owner=args.owner`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, which returns `"explicit"` when a non-empty owner was passed and `"default"` otherwise.
3. `sync_app/service.py::sync_item` passes only `effective_owner` into `sync_app/store.py::make_record`, so storage receives `owner` but not `owner_source` or `routing_key`.
4. `sync_app/service.py::sync_item` computes `routing_key` by calling `sync_app/serializer.py::build_routing_key(effective_owner, name)`.
5. `sync_app/service.py::sync_item` emits the final payload through `sync_app/serializer.py::serialize_payload(record, owner_source=owner_source, routing_key=routing_key)`, which appends those two fields onto a copy of the stored record.

## Why The Support Note Fails

- The storage-origin theory for `owner_source` is contradicted by `sync_app/store.py::make_record`, which stores only `name`, `status`, and `owner`, and by `sync_app/serializer.py::serialize_payload`, which is where `owner_source` is inserted into the outbound payload.
- The claim that `routing_key` is computed before `--owner` is applied is contradicted by `sync_app/cli.py::main` passing `owner=args.owner` into `sync_app/service.py::sync_item`, and by `sync_app/service.py::sync_item` computing `routing_key` only after `sync_app/service.py::_resolve_owner` returns the effective owner.
- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization`, `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`, and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` all assert the live payload values for `owner_source` and `routing_key`.

## Doc-Level Correction

The needed correction is documentation-only: update the explanation around the request path, not the storage layer. The current code already matches the concise statement in `docs/data_flow.md` that `sync_item` resolves the owner, persists the base record, and then emits the payload, which is consistent with `sync_app/service.py::sync_item`, `sync_app/store.py::make_record`, and `sync_app/serializer.py::serialize_payload`.
