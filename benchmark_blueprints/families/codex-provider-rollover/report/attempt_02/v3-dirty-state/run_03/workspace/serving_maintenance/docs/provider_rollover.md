# Provider rollover

The maintenance profile is pinned to `responses_proxy` and the responses endpoint `http://127.0.0.1:11434/v1/responses`.

Update the existing `serving_maintenance/.codex/config.toml` in place when rollover settings change. Do not regenerate the file from the stale template helper, and do not rewrite the local tuning block at the end of the file.

The profile must keep `wire_api = "responses"` and `store = true` so follow-up turns can send `previous_response_id` back to the responses endpoint unchanged.
