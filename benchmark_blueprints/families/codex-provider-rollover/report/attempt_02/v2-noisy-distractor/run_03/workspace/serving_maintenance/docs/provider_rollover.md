# Provider rollover

Keep the canonical maintenance profile on `responses_proxy`.

- Selected provider: `responses_proxy`
- Responses endpoint: `http://127.0.0.1:11434/v1/responses`
- Wire API: `responses`
- Store requirement: `store = true`

Ignore the stale rehearsal canary path. The maintenance profile should stay on the canonical proxy-backed responses endpoint for normal rollover validation.
