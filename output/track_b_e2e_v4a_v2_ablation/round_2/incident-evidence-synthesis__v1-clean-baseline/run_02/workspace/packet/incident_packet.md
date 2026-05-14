# Incident Packet: INC-2047

## Triggering Condition
Bulk-refund requests were accepted without idempotency keys at 10:17 UTC, leading to duplicate refunds for three enterprise accounts.

## Failed Guardrail
The idempotency-required guardrail was skipped due to the `legacy_batch_header` bypass, allowing non-idempotent bulk-refund requests to proceed.

## Highest-Confidence Follow-Up Action
Reject all bulk-refund requests that lack an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity
The initial theory of queue worker replay (TICKET-8721) was opened before API gateway logs were available and remains unconfirmed, though evidence from TICKET-8729 strongly points to the missing idempotency keys as the root cause.
