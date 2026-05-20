# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted at 10:17 UTC without idempotency keys, resulting in duplicate refunds for three enterprise accounts. Single-refund API traffic in the same window included idempotency keys and did not duplicate. The bulk-refund endpoint was disabled at 10:38 UTC.

Evidence: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`, `corpus/tickets/TICKET-8729.md`

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request carried a `legacy_batch_header`, which acted as a bypass. This allowed non-idempotent bulk-refund requests to be processed, directly causing the duplicate refunds.

Evidence: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`

## Follow-Up

1. Reject all bulk-refund requests that do not include an idempotency key.
2. Remove the `legacy_batch_header` bypass from the guardrail evaluation of the `idempotency-required` guardrail before re-enabling the endpoint.
3. Investigate worker replay and retry backoff as a secondary concerns.

Evidence: `corpus/remediation/notes.md`, `corpus/tickets/TICKET-8729.md`

## Ambiguity

TICKET-8721 initially attributed the duplicate refunds to a queue worker replaying an old payment event after a timeout. This theory was opened before API gateway logs were available and is not supported as the primary trigger by the final evidence. Whether the worker replay contributed to the incident is not confirmed, but the gateway logs clearly show the guardrail skip as the root cause.

Evidence: `corpus/tickets/TICKET-8721.md`, `corpus/logs/api_gateway_2026-05-01.log`
