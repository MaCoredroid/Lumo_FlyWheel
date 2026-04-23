# Smoke procedure

1. Run the first maintenance turn against `http://127.0.0.1:11434/v1/responses` with the `responses_proxy` profile and confirm the first response returns a completed `response.id`.
2. Run the follow-up turn with `store = true` and set `request.previous_response_id` to the exact `response.id` from turn one.
3. Verify exact continuity, not just presence: `turn2.request.previous_response_id` must equal `turn1.response.id`, and `turn2.response.previous_response_id` must also echo that same value.
4. Treat the smoke as failed if the second turn disables `store`, changes provider or base URL, omits `previous_response_id`, or returns a mismatched follow-up id.
