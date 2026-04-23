# Provider rollover

The maintenance cutover is for the `maintenance-responses` profile, not a generic Responses smoke.

That profile must select `responses_proxy` and send maintenance traffic through `http://127.0.0.1:11434/v1/responses`.
Leaving the config on `legacy_vllm` or the direct endpoint `http://127.0.0.1:8001/v1` bypasses the proxy-backed storage path that turn-two follow-ups need.

Treat the rollout as a profile repair:
- `profile = "maintenance-responses"`
- `provider = "responses_proxy"`
- `model_providers.responses_proxy.base_url = "http://127.0.0.1:11434/v1/responses"`
- `model_providers.responses_proxy.store = true`
