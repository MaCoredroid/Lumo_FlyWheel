# Incident Packet: INC-2047

## Triggering Condition
Bulk-refund requests were accepted without idempotency keys due to the `legacy_batch_header` bypass, resulting in duplicate refunds for three enterprise accounts.

## Failed Guardrail
The idempotency-required guardrail was skipped at 10:18 UTC when the `legacy_batch_header` was present, allowing duplicate refund requests to be processed.

## Highest-Confidence Follow-up Action
Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity
The initial theory of queue worker replay (TICKET-8721) was opened before API gateway logs were available and has been superseded by evidence from TICKET-8729 showing the duplicate refunds were caused by missing idempotency keys rather than worker replay.
