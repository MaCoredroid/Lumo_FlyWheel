# Remediation Notes

Early note: investigate worker replay and retry backoff.

Final note: bulk-refund requests must be rejected unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint.
