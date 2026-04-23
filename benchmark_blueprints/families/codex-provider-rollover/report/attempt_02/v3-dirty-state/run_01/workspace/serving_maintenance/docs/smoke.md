# Smoke procedure

1. Run the smoke against the maintenance profile and both fixture turns:
   `python3 serving_maintenance/scripts/smoke_responses_profile.py --config serving_maintenance/.codex/config.toml --turn1 serving_maintenance/fixtures/http/turn1_ok.json --turn2 serving_maintenance/fixtures/http/turn2_ok.json`
2. Confirm the selected provider is `responses_proxy` and the base URL is `http://127.0.0.1:11434/v1/responses`.
3. Confirm turn one completed with a response id.
4. Confirm the turn-two request carries `previous_response_id` exactly equal to the turn-one response id.
5. Confirm the turn-two response echoes the same `previous_response_id`, which proves exact follow-up continuity instead of a fresh unrelated request.
6. Confirm both turns keep `store = true`.
7. Treat any mismatch as a failure and rerun `bin/run-visible-tests` after the repair.
