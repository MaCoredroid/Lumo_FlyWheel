# Smoke procedure

1. Run the first maintenance turn against `http://127.0.0.1:11434/v1/responses` and confirm it returns a response id.
2. Run the second turn as a follow-up and require `request.previous_response_id` to exactly equal the first turn response id.
3. Also require the second response payload to echo that same `previous_response_id`, so the follow-up continuity is exact rather than merely present.
4. Keep `store=true` on the selected provider and on both requests; if storage drops, the continuity check is not valid.
