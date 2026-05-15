# Incident Packet: INC-2047

## Summary
Duplicate refunds were issued to enterprise accounts due to bulk-refund requests being accepted without idempotency keys, caused by a guardrail bypass.

## Triggering Condition
Bulk-refund requests were accepted without idempotency keys at 10:17 UTC. The `legacy_batch_header` flag` bypass allowed the `idempotency-required` guardrail to be skipped.

## Failed Guardrail
The `idempotency-required` guardrail was skipped due to the `legacy_batch_header` bypass mechanism: `legacy_batch_header`.

## Highest-Confidence Follow-Up Action
Reject bulk-refund requests that lack an idempotency key. Remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity
TICKET-8721 initially theorized queue worker replay as the cause. However, TICKET-8729 evidence shows all duplicates correlate with `request_class=bulk-refund` and missing idempotency keys, while single-refund traffic with idempotency keys did not duplicate. Worker replay remains unconfirmed but is less likely given the evidence.
