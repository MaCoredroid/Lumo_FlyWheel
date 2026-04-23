# Smoke procedure

1. Run the first maintenance turn against `http://127.0.0.1:11434/v1/responses` using the `responses_proxy` maintenance profile with `store = true`.
2. Record the turn-one response id from the first completed response.
3. Run the follow-up maintenance turn with `previous_response_id` set to that exact turn-one response id.
4. Verify that turn two keeps `store = true` and that `previous_response_id` exactly matches the turn-one response id in both the follow-up request payload and the follow-up response payload.
5. Treat any missing or mismatched `previous_response_id`, provider drift, base URL drift, or `store = false` value as a smoke failure.
