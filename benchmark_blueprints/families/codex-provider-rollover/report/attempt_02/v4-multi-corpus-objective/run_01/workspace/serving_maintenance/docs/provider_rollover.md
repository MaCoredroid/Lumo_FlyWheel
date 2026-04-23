# Provider rollover

The maintenance cutover is for the `maintenance-responses` profile, not a generic smoke-only repair.

That profile must select `responses_proxy` and target `http://127.0.0.1:11434/v1/responses` for proxy-backed Responses maintenance.

Keep the local tuning block in `serving_maintenance/.codex/config.toml` exactly as-is while updating the selected provider.
