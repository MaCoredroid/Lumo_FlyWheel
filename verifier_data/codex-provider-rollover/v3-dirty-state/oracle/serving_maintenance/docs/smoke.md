# Smoke procedure

1. Run the maintenance profile with `responses_proxy`.
2. Confirm the selected provider uses `http://127.0.0.1:11434/v1/responses`.
3. Run a second turn and require exact `previous_response_id` continuity.
4. Confirm the follow-up request keeps `store = true`.
