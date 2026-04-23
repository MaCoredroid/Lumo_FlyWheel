# Provider rollover

The earlier rollback was incomplete: it left the maintenance profile exposed to the `store = false` regression and only checked that a follow-up request carried some `previous_response_id`.

The active repair path is the `maintenance-responses` profile using `responses_proxy` at `http://127.0.0.1:11434/v1/responses`.

Keep the maintenance profile on the proxy endpoint during rollover recovery, and require follow-up validation that proves exact turn continuity instead of a generic smoke-only check.
