# Provider rollover

The maintained Responses profile must select `responses_proxy`, not `legacy_vllm`.

Required steady-state settings:

1. `provider = "responses_proxy"`
2. `model_providers.responses_proxy.base_url = "http://127.0.0.1:11434/v1/responses"`
3. `model_providers.responses_proxy.wire_api = "responses"`
4. `model_providers.responses_proxy.store = true`

Keep the local tuning block exactly as-is when you repair the profile.
