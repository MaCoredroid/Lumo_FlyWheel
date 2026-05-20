# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted by the API gateway without idempotency keys on 2026-05-01 at 10:17 UTC. This resulted in duplicate refunds for three enterprise accounts. The bulk-refund runbook was started at 10:15 UTC and requests were accepted without idempotency keys by 10:17 UTC.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log` (10:17:44Z and 10:17:46Z entries show `idempotency_key=missing` and `status=accepted`), `corpus/timeline/incident_timeline.md` (10:17 UTC entry), `corpus/tickets/TICKET-8729.md` (every duplicate has `request_class=bulk-refund` and no idempotency key).

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request carried a `legacy_batch_header`, which triggered by the `legacy_batch_header` bypass. This allowed duplicate bulk-refund requests to be accepted without deduplicate.

**Evidence:** `corpus/logs/api_gateway_2026-05-01.log` (10:18:02Z entry shows `guardrail=idempotency-required decision=skipped reason=legacy_batch_header`), `corpus/timeline/incident_timeline.md` (10:18 UTC entry).

## Follow-Up

The highest-confidence follow-up action is to reject all bulk-refund requests that do not include an idempotency key and to remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. The endpoint was disabled at 10:38 UTC.

**Evidence:** `corpus/remediation/notes.md` (final note), `corpus/timeline/incident_timeline.md` (10:38 UTC entry), `corpus/tickets/TICKET-8729.md` (single-refund API traffic in the same window included idempotency keys and did not duplicate).

## Ambiguity

An initial theory (TICKET-8721) attributed the duplicates to a queue worker replaying an old payment event after a timeout. However, final evidence from API gateway logs and TICKET-8729 does not support worker replay as the primary trigger; the root cause is the missing idempotency guardrail bypass. The worker replay theory remains unresolved but is low-confidence.

**Evidence:** `corpus/tickets/TICKET-8721.md` (initial theory noted as opened before API gateway logs were available), `corpus/tickets/TICKET-8729.md` (evidence points to idempotency key absence, not replay).
