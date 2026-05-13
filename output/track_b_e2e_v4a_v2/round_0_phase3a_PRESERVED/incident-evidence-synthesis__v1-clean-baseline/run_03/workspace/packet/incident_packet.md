# Incident Packet: INC-2047

## Triggering Condition
Bulk refund requests were accepted without idempotency keys at 10:17 UTC during the enterprise credits bulk-refund runbook execution.

## Failed Guardrail
The idempotency-required guardrail was bypassed at 10:18 UTC due to the `legacy_batch_header` flag being present in the request.

## Highest-Confidence Follow-Up Action
Reject all bulk-refund requests unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity
The initial theory (TICKET-8721) suggested a queue worker replayed an old payment event after a timeout. However, TICKET-8729 evidence indicates the root cause was the missing idempotency key validation for bulk-refund requests, not worker replay. The worker replay hypothesis should be formally dismissed.
