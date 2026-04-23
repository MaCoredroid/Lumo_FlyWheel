# Request Path Brief

        Verdict: the support note is incorrect.

        Live path:
        1. `sync_app/cli.py::main` parses `--owner` and forwards it into `sync_app/service.py::sync_item`.
        2. `sync_app/service.py::sync_item` calls `sync_app/service.py::_resolve_owner`, where the effective owner and `owner_source` are actually chosen.
        3. `sync_app/store.py::make_record` only persists the base record after owner selection; it does not decide `owner_source`.
        4. `sync_app/serializer.py::build_routing_key` derives `routing_key` from the resolved owner and item name.
        5. `sync_app/serializer.py::serialize_payload` is the emission step that exposes both `owner_source` and `routing_key`.

        Test evidence:
        - `tests/test_sync.py::test_cli_accepts_owner_flag_and_preserves_existing_fields`
        - `tests/test_sync.py::test_service_uses_default_owner_when_flag_is_missing`

        Rejected decoy:
        - `incident_context/prior_docs_correction.md::rolled_back_patch` is not live because rolled-back correction repeated the stale theory and was explicitly invalidated by incident follow-up.

Prior correction note: `incident_context/rollback_note.md::incident_follow_up` explains why the earlier store-layer explanation was rolled back.
