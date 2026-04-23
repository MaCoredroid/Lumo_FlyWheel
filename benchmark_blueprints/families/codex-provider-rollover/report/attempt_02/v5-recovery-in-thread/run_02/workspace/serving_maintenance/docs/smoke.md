# Smoke procedure

1. Run the first maintenance turn with the `maintenance-responses` profile on `responses_proxy` at `http://127.0.0.1:11434/v1/responses`.
2. Capture the first turn response id.
3. Run the second turn as a follow-up and require `store = true`.
4. Verify the second turn request `previous_response_id` exactly matches the first turn response id.
5. Verify the second turn response also reports the same `previous_response_id`, proving exact follow-up continuity rather than only the presence of a non-empty field.
