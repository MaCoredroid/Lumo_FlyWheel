# Smoke procedure

Run the maintenance-responses smoke as a two-turn continuity check, not a one-turn liveness probe.

1. Load `serving_maintenance/.codex/config.toml` and confirm the selected provider is `responses_proxy` with base URL `http://127.0.0.1:11434/v1/responses`, `wire_api = "responses"`, and `store = true`.
2. Run turn 1 and capture the exact response id from `response.id`.
3. Run turn 2 with `request.previous_response_id` set to that exact turn-1 id.
4. Verify `request.previous_response_id == turn1.response.id` exactly.
5. Verify `response.previous_response_id == turn1.response.id` exactly.
6. Fail the smoke if either turn sends `store = false`, if the provider/base URL drift, or if either follow-up id does not match the exact previous response id.
