# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys on 2026-05-01 starting at 10:17 UTC. The bulk-refund runbook was launched at 10:15 UTC for enterprise credits, and the API gateway accepted duplicate bulk-refund requests because the idempotency-required guardrail was skipped due to the `legacy_batch_header` bypass. This resulted in duplicate refunds for three enterprise accounts.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log` (10:17:44Z and 10:17:46Z entries show `idempotency_key=missing` and `duplicate_candidate=true`), `corpus/timeline/incident_timeline.md` (10:15–10:18 UTC entries), `corpus/tickets/TICKET-8729.md` (every duplicate has `request_class=bulk-refund` and no idempotency key).

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request carried the `legacy_batch_header`, which bypassed the idempotency check. Single-refund API traffic in the same window included idempotency keys and did not produce duplicates, confirming the guardrail works correctly when not bypassed.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log` (10:18:02Z entry: `guardrail=idempotency-required decision=skipped reason=legacy_batch_header`), `corpus/timeline/incident_timeline.md` (10:18 UTC entry), `corpus/tickets/TICKET-8729.md` (single-refund traffic unaffected).

## Follow-up

**Highest-confidence action:** Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

**Evidence:** `corpus/remediation/notes.md` (final note), `corpus/tickets/TICKET-8729.md` (evidence that only bulk-refund without idempotency keys duplicated).

## Ambiguity

An early theory (TICKET-8721) attributed the duplicates to a queue worker replaying an old payment event after a timeout. This theory was formed before API gateway logs were available and is not supported by the final evidence: every duplicate is a bulk-refund request missing an idempotency key, and single-refund traffic was unaffected. The worker-replay hypothesis remains unresolved but is not the primary trigger.

**Evidence:** `corpus/tickets/TICKET-8721.md` (initial theory predates log availability), `corpus/remediation/notes.md` (early note on worker replay investigation).
