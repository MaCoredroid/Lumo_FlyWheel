# Smoke procedure

Run the smoke against the `maintenance-responses` profile on `http://127.0.0.1:11434/v1/responses`.

1. Run turn one and confirm the request used `responses_proxy` with `store = true`.
2. Capture the exact turn-one response id.
3. Run turn two against the same route and require `request.previous_response_id` to equal the exact turn-one response id.
4. Require the turn-two response to report the same `previous_response_id`, proving exact follow-up continuity instead of just a non-empty field.
5. Fail the smoke if either turn drops `store = true`, because one-turn success can still hide the maintenance regression that breaks follow-up retrieval.
