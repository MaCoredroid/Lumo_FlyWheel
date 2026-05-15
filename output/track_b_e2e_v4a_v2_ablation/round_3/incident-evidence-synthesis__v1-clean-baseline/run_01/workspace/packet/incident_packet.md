# Incident Packet: INC-2047

## Triggering Condition
Bulk-refund requests were accepted without idempotency keys, leading to duplicate refunds for three enterprise accounts.

## Failed Guardrail
The idempotency-required guardrail was skipped due to the `legacy_batch_header` bypass, allowing duplicate refund requests to process.

## Highest-Confidence Follow-Up Action
Reject bulk-refund requests unless they include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity
Worker replay was initially theorized as the cause (TICKET-8721), but final evidence (TICKET-8729) indicates this is not the primary trigger. The worker replay theory remains unsupported by the final evidence.
