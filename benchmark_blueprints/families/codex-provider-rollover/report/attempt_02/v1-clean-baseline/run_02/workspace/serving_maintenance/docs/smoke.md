# Smoke procedure

Run the smoke with the maintained profile and the two captured HTTP turns.

Expected checks:
1. The selected provider is `responses_proxy`.
2. The provider base URL is `http://127.0.0.1:11434/v1/responses`.
3. Both turn requests use `store = true`.
4. Turn 1 returns a response id.
5. Turn 2 sends `previous_response_id` exactly equal to the turn 1 response id.
6. Turn 2 echoes that same `previous_response_id` in the completed response.

Any mismatch in provider selection, proxy URL, `store`, or exact `previous_response_id` continuity should fail the smoke.
