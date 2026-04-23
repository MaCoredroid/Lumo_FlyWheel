# Smoke procedure

1. Run turn one with the `maintenance-responses` profile through `responses_proxy` at `http://127.0.0.1:11434/v1/responses` and confirm the first response returns an id.
2. Run turn two against the same profile and require `store=true` on both turns so follow-up retrieval remains available.
3. Verify exact continuity: turn two must send the turn-one id as `previous_response_id`, and the follow-up response must report that same `previous_response_id` back.
4. Treat any missing or mismatched `previous_response_id` as a failed rollover check, even if the first turn succeeded.
