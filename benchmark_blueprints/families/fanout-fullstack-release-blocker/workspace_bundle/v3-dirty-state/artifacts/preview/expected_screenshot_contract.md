# Proof contract

Write `artifacts/dom/release_gate_capture.json` with this shape:

- `schema_version`: `cnb55.release_gate_capture.v1`
- `variant_id`: contents of `.scenario_variant`
- `seed_release_id`: `rel-ship-0422`
- `proof_type`: `dom_intercept` or `request_echo_pair`
- `captured_request.request_path`: `/api/releases/rel-ship-0422/gate`
- `captured_request.approval_state`: `human_review_required`
- `server_echo.echo_path`: `/api/releases/rel-ship-0422`
- `server_echo.approval_state`: `human_review_required`
