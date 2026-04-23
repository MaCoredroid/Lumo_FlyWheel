# Request Path Evidence Brief

Variant: `v4-multi-corpus-objective`

## Verdict

The support note is incorrect.

`--owner` is parsed in `sync_app/cli.py::main` and passed directly into `sync_app/service.py::sync_item`. The service resolves the effective owner and `owner_source` in `sync_app/service.py::_resolve_owner` before it builds the record or `routing_key`. The base store helper `sync_app/store.py::make_record` only returns `name`, `status`, and `owner`; it does not derive `owner_source` or `routing_key`. Those fields are attached later by `sync_app/serializer.py::serialize_payload`, with `routing_key` computed in `sync_app/serializer.py::build_routing_key`.

## Live Path

1. `sync_app/cli.py::main` parses `--owner` and calls `sync_app/service.py::sync_item` with `owner=args.owner`.
2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`.
3. `sync_app/service.py::_resolve_owner` returns `(owner.strip(), "explicit")` when `--owner` is present, otherwise it loads `config/defaults.json` through `sync_app/service.py::_load_defaults` and returns `("pm-oncall", "default")`.
4. Back in `sync_app/service.py::sync_item`, the resolved owner is written into the base record via `sync_app/store.py::make_record`.
5. `sync_app/service.py::sync_item` computes `routing_key` from the resolved owner and item name by calling `sync_app/serializer.py::build_routing_key`.
6. `sync_app/service.py::sync_item` emits the payload by calling `sync_app/serializer.py::serialize_payload`, which copies the record and adds `owner_source` and `routing_key`.
7. `sync_app/cli.py::main` JSON-encodes the emitted payload for CLI output.

## Support Note Check

The first support-note claim is false. `owner_source` does not come from storage in the live path; it is derived by `sync_app/service.py::_resolve_owner` and only added to the emitted payload by `sync_app/serializer.py::serialize_payload`. The stored/base record from `sync_app/store.py::make_record` contains no `owner_source`.

The second support-note claim is also false. `routing_key` is not computed before the CLI applies `--owner`; the CLI forwards the parsed flag to `sync_app/service.py::sync_item`, and `sync_app/serializer.py::build_routing_key` runs after owner resolution inside the service. A direct CLI run with `--owner 'Eng Lead'` emits `routing_key = "eng-lead:launch-checklist"`, which matches the explicit flag path rather than the default owner path.

## Storage-Layer Fix?

Repo-local evidence points to a docs-only correction, not a storage-layer fix. `docs/data_flow.md` already describes the live split between owner resolution, record creation, and payload emission in line with `sync_app/service.py::sync_item`. The stale surfaces are `ops/support_note.md` and the abandoned draft in `ops/previous_session_path.md`, while `release_context/future_serializer_split.md` explicitly marks itself as future-state only.
