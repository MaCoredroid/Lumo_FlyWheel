# Provider rollover

Keep the canonical maintenance profile selected:
- provider: `responses_proxy`
- endpoint: `http://127.0.0.1:11434/v1/responses`
- wire API: `responses`
- store: `true`

Ignore the stale canary path and do not switch maintenance traffic to `maintenance_canary`.
When repairing the profile, preserve the local tuning block at the end of `serving_maintenance/.codex/config.toml` exactly as-is.
