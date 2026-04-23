# Provider rollover

The maintenance profile must select `responses_proxy` and send Responses API traffic to `http://127.0.0.1:11434/v1/responses`.

Update only the provider selection in `serving_maintenance/.codex/config.toml`.
Do not regenerate the file from `serving_maintenance/templates/legacy_profile_template.toml`, and do not rewrite the whole TOML from the stale helper.

Keep the local tuning block at the end of the file exactly as written:

```toml
# Preserve this tuning block byte-for-byte.
max_output_tokens = 6400
reasoning_summary = "auto"
tool_retry_budget = 3
proxy_read_timeout_ms = 9000
```

The selected provider must keep `wire_api = "responses"` and `store = true` so follow-up turns can reuse the exact `previous_response_id` from the prior response.
