# Incident Packet: INC-2047

## Triggering Condition
Bulk refund requests were accepted without idempotency keys, causing duplicate refunds for three enterprise accounts.

## Failed Guardrail
The idempotency-required guardrail was skipped due to the `legacy_batch_header` bypass being active on the bulk-refund endpoint.

## Highest-Confidence Follow-Up Action
Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the endpoint.

## Unresolved Ambiguity
The initial theory in TICKET-8721 suggested a queue worker replay caused the duplicates, but this was opened before API gateway logs were available. Evidence from TICKET-8729 indicates the root cause was the missing idempotency keys, not worker replay.
