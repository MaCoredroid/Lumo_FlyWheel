# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys during the bulk-refund runbook started at 10:15 UTC, bulk-refund requests were accepted by the API gateway without idempotency keys. Gateway logs show two consecutive bulk-refund requests at 10:17:44 and 10:17:46 with `idempotency_key=missing` and `status=accepted`, one flagged as `duplicate_candidate=true`. Duplicate refunds were reported at 10:25 UTC across three enterprise accounts.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log` (10:17 entries), `corpus/timeline/incident_timeline.md` (10:15–10:25), `corpus/tickets/TICKET-8729.md`

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18:02 UTC because the request carried a `legacy_batch_header`, which triggered by the `legacy_batch_header` bypass. This allowed duplicate bulk-refund requests to be processed. Single-refund API traffic in the same window included idempotency keys and did not duplicate, confirming the guardrail skip was the differentiating factor.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log` (10:18:02 entry), `corpus/timeline/incident_timeline.md` (10:18), `corpus/tickets/TICKET-8729.md`

## Follow-Up

Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass from the guardrail before re-enabling the bulk-refund endpoint. This is the highest-confidence action, directly supported by the remediation notes and the evidence that every duplicate refund lacked an idempotency key.

**Evidence:** `corpus/remediation/notes.md` (final note), `corpus/tickets/TICKET-8729.md`

## Ambiguity

TICKET-8721 initially theorized that a queue worker replayed an old payment event after a timeout. This theory was formed before API gateway logs were available and is not supported as the primary trigger by the final evidence. The extent to which worker replay contributed to the incident remains unclear, though the gateway-level evidence points to the guardrail skip as the root cause.

**Evidence:** `corpus/tickets/TICKET-8721.md`
