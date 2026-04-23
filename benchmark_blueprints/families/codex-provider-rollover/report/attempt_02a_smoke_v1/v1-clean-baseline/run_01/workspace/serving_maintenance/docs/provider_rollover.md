# Provider rollover

The maintained Responses profile must select `responses_proxy`.

- `provider` must be `responses_proxy`.
- `model_providers.responses_proxy.base_url` must remain `http://127.0.0.1:11434/v1/responses`.
- `wire_api` must stay `responses`.
- `store` must stay `true` so the follow-up turn can reuse the exact prior response id.

Do not rewrite `serving_maintenance/.codex/config.toml` from a template. Preserve the local tuning block exactly as written and only make the minimal profile repair needed for rollover.
