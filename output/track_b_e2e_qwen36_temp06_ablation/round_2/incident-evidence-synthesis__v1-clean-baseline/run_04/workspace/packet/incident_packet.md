# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys on 2026-05-01 starting at 10:17 UTC, resulting in duplicate refunds for three enterprise accounts. The bulk-refund runbook was started at 10:15 UTC and requests began at 10:15 UTC for enterprise credits. The API gateway logs show `idempotency_key=missing` and `status=accepted` for `request_class=bulk-refund` requests. Single-refund API traffic in the same window included idempotency keys and did not duplicate.

**Evidence**: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/tickets/TICKET-8729.md`, `corpus/timeline/incident_timeline.md`

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because of the `legacy_batch_header` bypass. This allowed duplicate bulk-refund requests to be processed without deduplication. The guardrail skip is recorded in the API gateway log with `decision=skipped` and `reason=legacy_batch_header`.

**Evidence**: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`

## Follow-up

Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. This is the highest-confidence action supported by the final remediation notes.

**Evidence**: `corpus/remediation/notes.md`, `corpus/tickets/TICKET-8729.md`

## Ambiguity

An early theory in TICKET-8721 attributed the duplicates to a queue worker replaying an old payment event after a timeout. This theory was opened before API gateway logs were available and is not supported as the primary trigger by the final evidence. The worker-replay hypothesis remains unconfirmed but is considered secondary to the idempotency guardrail bypass.

**Evidence**: `corpus/tickets/TICKET-8721.md`, `corpus/remediation/notes.md`
