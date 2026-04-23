# Request Path Brief

Support note verdict: the support note is not correct.

The live path starts at `sync_app/cli.py::main`, where `--owner` is parsed and passed directly into `sync_app/service.py::sync_item`. Inside `sync_app/service.py::sync_item`, the repo resolves the effective owner and `owner_source` first via `sync_app/service.py::_resolve_owner`, then builds the base record with `sync_app/store.py::make_record`, then derives `routing_key` from the resolved owner and item name with `sync_app/serializer.py::build_routing_key`, and only then emits the final payload through `sync_app/serializer.py::serialize_payload`.

That means `owner_source` does not come from storage. The only live base-record builder, `sync_app/store.py::make_record`, returns just `name`, `status`, and `owner`; it does not add `owner_source`. The emitted `owner_source` field is attached later by `sync_app/serializer.py::serialize_payload`, using the value returned by `sync_app/service.py::_resolve_owner`. The storage-layer theory in `ops/support_note.md` is contradicted by `sync_app/service.py::_resolve_owner`, `sync_app/store.py::make_record`, and `sync_app/serializer.py::serialize_payload`.

The note is also wrong about `routing_key`. The CLI does apply `--owner` before `routing_key` is computed because `sync_app/cli.py::main` passes `args.owner` into `sync_app/service.py::sync_item`, and `sync_app/service.py::sync_item` calls `sync_app/serializer.py::build_routing_key` with `effective_owner` after `sync_app/service.py::_resolve_owner` has already consumed that CLI input.

Repo-local tests line up with the implementation. `tests/test_sync.py::test_service_resolves_explicit_owner_before_serialization` expects `owner_source == "explicit"` and `routing_key == "pm-oncall:launch-checklist"` when `owner="pm-oncall"` is supplied. `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing` expects `owner_source == "default"` and derives the owner from `config/defaults.json`. `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields` confirms the CLI path preserves the same emitted fields through `sync_app/cli.py::main`.

Conclusion: the fix belongs in docs only, not in the storage layer. A correct note should say that the live derivation happens in `sync_app/service.py::sync_item` and `sync_app/serializer.py::serialize_payload`, while `sync_app/store.py::make_record` only carries the already-resolved owner field.
