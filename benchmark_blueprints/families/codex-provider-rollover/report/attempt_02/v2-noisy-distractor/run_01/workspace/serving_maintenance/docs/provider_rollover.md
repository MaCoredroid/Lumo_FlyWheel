# Provider rollover

Keep the canonical maintenance profile on `responses_proxy`.
The selected provider must resolve to `http://127.0.0.1:11434/v1/responses` with `wire_api = "responses"` and `store = true`.

Ignore the stale `maintenance_canary` path during maintenance rehearsals.
Do not replace the whole config from a template; update only the selected profile settings and preserve the local tuning block exactly.
