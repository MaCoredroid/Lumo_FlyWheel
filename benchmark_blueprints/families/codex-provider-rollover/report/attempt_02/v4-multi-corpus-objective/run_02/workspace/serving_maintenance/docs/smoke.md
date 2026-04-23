# Smoke procedure

1. Run the first maintenance turn with the `maintenance-responses` profile and confirm it is using `responses_proxy` on `http://127.0.0.1:11434/v1/responses`.
2. Record the exact turn-one response id.
3. Run the second turn as a follow-up and verify the request sends that exact id as `previous_response_id`.
4. Verify the follow-up response also reports the same `previous_response_id`, which confirms retrieval continuity instead of a one-turn-only smoke pass.
5. Confirm both turns were sent with `store = true`; stop only after the exact follow-up continuity check passes.
