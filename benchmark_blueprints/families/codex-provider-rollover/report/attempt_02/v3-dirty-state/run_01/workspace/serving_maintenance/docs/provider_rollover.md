# Provider rollover

The maintenance profile must select `responses_proxy` and route traffic through `http://127.0.0.1:11434/v1/responses`.

Update only the provider selection in `serving_maintenance/.codex/config.toml`.
Do not regenerate the whole file from `serving_maintenance/templates/legacy_profile_template.toml`.
Preserve the local tuning block exactly as written.

After the rollover, verify a two-turn follow-up path with the smoke script so the second request carries the exact `previous_response_id` from turn one and keeps `store = true`.
