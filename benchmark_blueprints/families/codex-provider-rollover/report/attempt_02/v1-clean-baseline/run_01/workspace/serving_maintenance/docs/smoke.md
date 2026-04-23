# Smoke procedure

1. Run a first maintenance turn through `http://127.0.0.1:11434/v1/responses` with `provider = "responses_proxy"` and `store = true`.
2. Capture `turn1.response.id`.
3. Run a second maintenance turn with `request.previous_response_id` set to that exact turn-1 id.
4. Verify `turn2.request.previous_response_id == turn1.response.id`.
5. Verify `turn2.response.previous_response_id == turn1.response.id`.
6. Fail the smoke if provider selection, proxy base URL, `store`, or either `previous_response_id` check drifts.
