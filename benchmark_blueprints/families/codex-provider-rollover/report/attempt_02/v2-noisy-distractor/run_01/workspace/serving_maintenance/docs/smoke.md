# Smoke procedure

1. Run the first maintenance turn against `http://127.0.0.1:11434/v1/responses` and confirm the response returns a concrete id.
2. Run the follow-up turn with `store = true` and `previous_response_id` set to the exact id returned by turn one.
3. Verify the follow-up request and the follow-up response both carry the exact previous_response_id from turn one before declaring continuity healthy.
