# Request Path Evidence Brief

## Verdict

The support note is **not correct** as stated. Repo-local code shows that `owner_source` is decided in `sync_app/service.py::_resolve_owner`, `routing_key` is computed in `sync_app/serializer.py::build_routing_key` after the effective owner has already been resolved inside `sync_app/service.py::sync_item`, and both fields are attached only at payload emission time by `sync_app/serializer.py::serialize_payload`.

## Live Path

The live request path starts at `sync_app/cli.py::main`, where `--owner` is parsed and forwarded as `owner=args.owner` into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the code first calls `sync_app/service.py::_resolve_owner`, which returns both the effective owner string and the `owner_source` label. Only after that does the service create the base record with `sync_app/store.py::make_record`, compute `routing_key` with `sync_app/serializer.py::build_routing_key`, and emit the final payload with `sync_app/serializer.py::serialize_payload`.

That ordering matters. `sync_app/store.py::make_record` stores only `name`, `status`, and `owner`; it does not persist `owner_source` or `routing_key`. The final payload gains those two fields only when `sync_app/serializer.py::serialize_payload` copies the record and adds them explicitly. This matches `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` and `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`, which assert that an explicit `--owner` yields `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"`.

## Support Note Check

The first support-note claim is unsupported by the live code. The presence of `owner` in the stored record does not imply that `owner_source` comes from storage, because storage never writes `owner_source` on the live path. The relevant evidence is `sync_app/store.py::make_record`, `sync_app/service.py::sync_item`, and `sync_app/serializer.py::serialize_payload`.

The second support-note claim is also incorrect. `routing_key` is not computed before the CLI applies `--owner`; the CLI passes the parsed flag into `sync_app/service.py::sync_item`, then the service resolves the owner first and computes `routing_key` from that resolved owner. The relevant evidence is `sync_app/cli.py::main`, `sync_app/service.py::_resolve_owner`, `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key`.

## Observations

The current default owner in `config/defaults.json` is also `pm-oncall`, so an explicit `--owner pm-oncall` and an omitted `--owner` can produce the same visible `owner` and `routing_key` values while still differing in `owner_source`. That makes the support-note intuition understandable, but the code and tests still show that the decision point lives in service/serializer logic, not in storage. This is reinforced by `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` and by `docs/data_flow.md`, especially the note tied to `docs/data_flow.md`'s discussion of over-compressed historical flow descriptions.
