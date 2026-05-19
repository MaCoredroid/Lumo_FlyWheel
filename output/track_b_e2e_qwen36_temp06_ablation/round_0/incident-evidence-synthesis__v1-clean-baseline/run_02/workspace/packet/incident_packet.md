# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys during the enterprise credits runbook on 2026-05-01 at 10:17 UTC. Gateway logs show two consecutive `request_class=bulk-refund` entries with `idempotency_key=missing` and `status=accepted`, the second flagged `duplicate_candidate=true`.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`, `corpus/tickets/TICKET-8729.md`

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request carried a `legacy_batch_header`, which the guardrail treated as a bypass. Single-refund API traffic in the same window included idempotency keys and did not duplicate, confirming the guardrail skip as the differentiator.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`, `corpus/tickets/TICKET-8729.md`

## Follow-up

Reject all bulk-refund requests that lack an idempotency key and remove the `legacy_batch_header` bypass from the guardrail before re-enabling the bulk-refund endpoint. This is the highest-confidence action supported by all final evidence.

**Evidence:** `corpus/remediation/notes.md`, `corpus/tickets/TICKET-8729.md`

## Ambiguity

An early theory (TICKET-8721) attributed the duplicates to a queue worker replaying an old payment event after a timeout. This theory was formed before API gateway logs were available and is not supported as the primary trigger by the final evidence. The exact scope of accounts affected beyond the three reported enterprise accounts is not fully quantified.

**Evidence:** `corpus/tickets/TICKET-8721.md`, `corpus/remediation/notes.md`
