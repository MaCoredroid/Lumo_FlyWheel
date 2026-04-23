# Request Path Evidence Brief

## Verdict

The support note is **not correct**. In the live path, `owner_source` is decided in `sync_app/service.py::_resolve_owner`, not recovered from storage, and `routing_key` is computed in `sync_app/serializer.py::build_routing_key` after `sync_app/service.py::_resolve_owner` has already produced the effective owner that came from `--owner` or defaults. The orchestrating call site is `sync_app/service.py::sync_item`, and the CLI only forwards the raw flag value through `sync_app/cli.py::main`.

## Live Request Path

The request path is:

`sync_app/cli.py::main` -> `sync_app/service.py::sync_item` -> `sync_app/service.py::_resolve_owner` -> `sync_app/store.py::make_record` -> `sync_app/serializer.py::build_routing_key` -> `sync_app/serializer.py::serialize_payload`

What each step does:

- `sync_app/cli.py::main` parses `--owner` and passes `args.owner` into `sync_app/service.py::sync_item`.
- `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when a non-empty owner flag is present; otherwise it loads `config/defaults.json` and returns `(defaults["owner"], "default")`.
- `sync_app/store.py::make_record` persists only the base record fields `name`, `status`, and `owner`. It does not assign `owner_source` or `routing_key`.
- `sync_app/serializer.py::build_routing_key` derives `routing_key` from the already-resolved owner plus `name`.
- `sync_app/serializer.py::serialize_payload` copies the base record and appends `owner_source` and `routing_key` into the emitted payload.

## Why The Support Note Fails

The first support-note claim says `owner_source` "looks like it comes from storage." That inference is contradicted by the live code. `sync_app/store.py::make_record` only returns `{"name", "status", "owner"}`, while `sync_app/serializer.py::serialize_payload` appends `owner_source` from the service-layer argument supplied by `sync_app/service.py::sync_item`. The only storage-adjacent helper that could draft `owner_source` from a record is `sync_app/serializer.py::draft_owner_source_from_record`, but that helper is not called from the live path.

The second support-note claim says `routing_key` is "probably computed before the CLI applies `--owner`." The live call graph shows the opposite: `sync_app/cli.py::main` passes `args.owner` into `sync_app/service.py::sync_item`, `sync_app/service.py::_resolve_owner` computes the effective owner, and only then does `sync_app/serializer.py::build_routing_key` run. The service-level tests in `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` explicitly assert the emitted `routing_key` for an explicit owner path.

## Observed Emission

Direct CLI observation matches the code and tests:

- With `--owner pm-oncall`, the emitted payload contains `owner_source: "explicit"` and `routing_key: "pm-oncall:launch-checklist"`, matching `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`.
- Without `--owner`, the emitted payload contains `owner_source: "default"` and still emits `routing_key: "pm-oncall:launch-checklist"` because `config/defaults.json` currently sets the default owner to `pm-oncall`, as covered by `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`.

That shared `routing_key` value is caused by the current default matching the explicit example, not by pre-CLI routing computation. `config/defaults.json` and `sync_app/service.py::_resolve_owner` explain the collision.

## Conclusion

The needed correction is documentation-level, not a storage-layer behavioral fix. `docs/data_flow.md::Data Flow Note` is aligned with the implementation: the service resolves the effective owner, persists the base record, and then emits the payload. The misleading artifact is `ops/support_note.md::Escalation Summary`, whose two speculative bullets should be replaced with the service-layer explanation above.
