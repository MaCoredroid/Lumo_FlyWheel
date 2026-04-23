# Smoke procedure

Run the maintenance smoke against a stored first turn and an exact follow-up turn.

1. Load the maintained config and confirm it selects `responses_proxy` at `http://127.0.0.1:11434/v1/responses`.
2. Confirm the provider and both recorded requests keep `store=true`.
3. Read the first turn response id.
4. Verify `turn2.request.previous_response_id` exactly equals the first turn response id.
5. Verify `turn2.response.previous_response_id` exactly equals the same first turn response id.
6. Fail the smoke if any provider, store, or `previous_response_id` continuity check drifts.

The smoke should not stop after merely seeing a response id. It only passes when the second turn proves exact follow-up continuity through `previous_response_id`.
