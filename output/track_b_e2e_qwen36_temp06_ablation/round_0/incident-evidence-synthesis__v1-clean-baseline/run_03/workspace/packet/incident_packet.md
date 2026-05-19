# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys, causing duplicate refunds for three enterprise accounts. The bulk-refund runbook started at 10:15 UTC, and by 10:17 UTC the API gateway was accepting bulk-refund requests with `idempotency_key=missing`. Every duplicate refund carried `request_class=bulk-refund`; single-refund traffic in the same window included idempotency keys and did not duplicate.

Evidence: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`, `corpus/tickets/TICKET-8729.md`

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request matched the `legacy_batch_header` bypass. This allowed non-idempotent bulk-refund requests to be accepted without replay protection.

Evidence: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`

## Follow-up

Reject all bulk-refund requests that lack an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. This is the highest-confidence corrective action supported by the final evidence.

Evidence: `corpus/remediation/notes.md`, `corpus/tickets/TICKET-8729.md`

## Ambiguity

An early theory (TICKET-8721) attributed the duplicates to a queue worker replaying an old payment event after a timeout. That theory was opened before API gateway logs were available and is not supported as the primary trigger by the final evidence. The worker-replay hypothesis remains unconfirmed but is superseded by the idempotency-guardrail finding.

Evidence: `corpus/tickets/TICKET-8721.md`, `corpus/remediation/notes.md`
