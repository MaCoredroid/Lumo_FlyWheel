# Smoke procedure

1. Run the first maintenance turn with the `responses_proxy` profile against `http://127.0.0.1:11434/v1/responses`.
2. Confirm turn one persisted state by checking that `store` is `true` and the first response includes an id.
3. Run the follow-up turn with `previous_response_id` set to the exact turn-one response id.
4. Verify exact continuity on both sides of the follow-up:
   `turn2.request.previous_response_id == turn1.response.id`
   `turn2.response.previous_response_id == turn1.response.id`
5. Fail the smoke if the provider, endpoint, `store`, or exact `previous_response_id` continuity drifts.
