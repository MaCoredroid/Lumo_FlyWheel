# Incident Packet: INC-2047

## Triggering Condition
Bulk refund requests were accepted without idempotency keys when the `legacy_batch_header` was present, allowing duplicate refunds to be processed.

## Failed Guardrail
The idempotency-required guardrail was bypassed at 10:18 UTC due to the `legacy_batch_header` flag being honored. This allowed non-idempotent bulk refund operations to proceed.

## Highest-Confidence Follow-Up Action
Reject all bulk-refund requests that do not include an idempotency key. Remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity
The initial theory (TICKET-8721) that a queue worker replayed an old payment event was opened before API gateway logs were available. This theory remains unverified, though TICKET-8729 provides stronger evidence pointing to the missing idempotency keys as the root cause.
