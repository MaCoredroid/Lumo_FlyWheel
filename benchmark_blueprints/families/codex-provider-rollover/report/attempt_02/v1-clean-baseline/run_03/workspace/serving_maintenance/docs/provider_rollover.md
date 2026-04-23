# Provider rollover

The maintained Responses profile is `responses_proxy`.

- Selected provider: `responses_proxy`
- Endpoint: `http://127.0.0.1:11434/v1/responses`
- Wire API: `responses`
- Store requirement: `true`

Keep the local tuning block in `serving_maintenance/.codex/config.toml` exactly as-is during rollover work. Only repair the maintained provider selection or provider-specific fields that drift away from the proxy-backed Responses profile.
