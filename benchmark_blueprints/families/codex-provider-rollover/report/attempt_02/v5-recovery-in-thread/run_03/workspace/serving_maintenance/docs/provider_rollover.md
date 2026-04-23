# Provider rollover

The maintenance profile now selects `responses_proxy` and targets `http://127.0.0.1:11434/v1/responses`.

This repair explicitly addresses the earlier rollback: the selected provider path had `store = false`, and the follow-up smoke only checked that a `previous_response_id` field existed instead of verifying exact continuity against the prior response id.

For maintenance runs, keep the profile on `maintenance-responses` with the proxy-backed Responses endpoint and `store = true` on the selected provider so turn-two follow-ups can resume from the exact prior response state.
