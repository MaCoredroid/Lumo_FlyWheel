# Provider rollover

The maintained Responses profile must select `responses_proxy`.

Required profile shape:
- `provider = "responses_proxy"`
- `model_providers.responses_proxy.base_url = "http://127.0.0.1:11434/v1/responses"`
- `model_providers.responses_proxy.wire_api = "responses"`
- `model_providers.responses_proxy.store = true`

Keep the local tuning block exactly as written when repairing the profile. Do not rewrite the TOML from a template.
