# Smoke procedure

1. Run the first maintenance turn through `responses_proxy` at `http://127.0.0.1:11434/v1/responses` with `store=true`.
2. Record the first turn `response.id`.
3. Run the follow-up turn with `previous_response_id` set to that exact first-turn id.
4. Verify the second request keeps `store=true`.
5. Verify the second response also echoes the same `previous_response_id` value, not just any non-empty continuity token.
6. Treat the smoke as failed if either request-side or response-side continuity diverges from the exact first-turn id.
