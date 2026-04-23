# Request Path Brief

## Verdict

The support note is not correct. `owner_source` is decided in `sync_app/service.py::_resolve_owner`, not in storage, and `routing_key` is computed in `sync_app/serializer.py::build_routing_key` only after `sync_app/cli.py::main` has forwarded `--owner` into `sync_app/service.py::sync_item`. The live evidence supports a docs-only correction, not a storage-layer code fix.

## Live Request Path

1. `sync_app/cli.py::main` parses `--owner` and forwards `args.owner` into `sync_app/service.py::sync_item`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner` to derive both the effective `owner` and `owner_source`.
3. `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when the CLI provided a non-blank owner, otherwise it loads the fallback owner through `sync_app/service.py::_load_defaults` from `config/defaults.json::__file__` and returns `"default"`.
4. `sync_app/store.py::make_record` persists only the base record fields `name`, `status`, and `owner`; it does not derive or attach `owner_source` or `routing_key`.
5. Back in `sync_app/service.py::sync_item`, the already-resolved owner is passed to `sync_app/serializer.py::build_routing_key`, so `routing_key` is built after owner precedence is settled.
6. `sync_app/serializer.py::serialize_payload` copies the base record and adds the derived `owner_source` and `routing_key`, and `sync_app/cli.py::main` emits that payload as JSON.

## Why The Support Note Fails

- The storage-layer theory in `ops/support_note.md::Escalation Summary` is contradicted by `sync_app/store.py::make_record`, which returns only `name`, `status`, and `owner`. The only storage helper that can attach `owner_source` is `sync_app/store.py::legacy_make_record_with_owner_source`, and repo-local call-site search shows it is unused in the live path.
- The claim that `routing_key` is computed before CLI owner precedence is contradicted by `sync_app/cli.py::main` calling `sync_app/service.py::sync_item`, then `sync_app/service.py::sync_item` resolving owner first via `sync_app/service.py::_resolve_owner`, and only then invoking `sync_app/serializer.py::build_routing_key`.
- The serializer-planning note `release_context/future_serializer_split.md::Planned Serializer Split` is explicitly future intent, not proof that current owner attribution moved elsewhere.

## Test And Runtime Evidence

- `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` proves the explicit-owner path emits `owner="pm-oncall"`, `owner_source="explicit"`, and `routing_key="pm-oncall:launch-checklist"`.
- `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` proves the fallback path emits `owner_source="default"` while sourcing the owner from `config/defaults.json::__file__`.
- `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` proves the CLI path preserves the explicit owner all the way through emitted JSON.
- A direct CLI run through `sync_app/cli.py::main` emitted `{"name": "Launch Checklist", "owner": "pm-oncall", "owner_source": "explicit", "routing_key": "pm-oncall:launch-checklist", "status": "pending"}`, matching the live code path above.
