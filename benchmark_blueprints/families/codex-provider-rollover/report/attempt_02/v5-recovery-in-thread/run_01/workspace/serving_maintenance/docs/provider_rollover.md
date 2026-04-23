# Provider rollover

The earlier rollback came from two gaps in the maintenance-responses profile:

1. The selected provider drifted back to `legacy_vllm` instead of `responses_proxy`.
2. Follow-up validation stopped after a first successful turn, so `store = false` and broken `previous_response_id` continuity were not caught before rollback.

The repaired profile must keep maintenance traffic on:

- provider: `responses_proxy`
- base URL: `http://127.0.0.1:11434/v1/responses`
- wire API: `responses`
- store: `true`

Do not rewrite the file from a template. Update only the profile selection and preserve the local tuning block exactly as written.
