# Smoke procedure

1. Run turn one against `responses_proxy` at `http://127.0.0.1:11434/v1/responses` with `store = true` and capture the response id.
2. Run turn two against the same endpoint with `store = true`.
3. Verify the turn two request sets `previous_response_id` to the exact turn one response id.
4. Verify the turn two response echoes the same `previous_response_id` and finishes with `status = completed`.
5. Fail the smoke on any provider, base URL, `store`, or `previous_response_id` mismatch.
