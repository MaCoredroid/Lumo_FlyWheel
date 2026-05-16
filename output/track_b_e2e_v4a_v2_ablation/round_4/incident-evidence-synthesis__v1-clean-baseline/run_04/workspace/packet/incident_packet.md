# Incident Packet: INC-2047

## Triggering Condition
Bulk refund requests were accepted without idempotency keys at 10:17 UTC when the `legacy_batch_header` bypass was active, causing duplicate refunds for enterprise accounts.

## Failed Guardrail
The idempotency-required guardrail was skipped at 10:18 UTC due to the `legacy_batch_header` bypass.

## Highest-Confidence Follow-Up Action
Reject bulk-refund requests unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint.

## Unresolved Ambiguity
Initial theory in TICKET-8721 suggested queue worker replay caused duplicates, but this was formed before API gateway logs were available. TICKET-8729 provides stronger evidence pointing to missing idempotency keys as the root cause.
