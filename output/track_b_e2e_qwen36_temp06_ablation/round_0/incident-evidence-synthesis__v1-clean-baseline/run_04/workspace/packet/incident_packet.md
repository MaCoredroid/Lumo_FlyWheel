# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys during a bulk-refund runbook executed on 2026-05-01. At 10:17 UTC, the API gateway logged `request_class=bulk-refund` requests with `idempotency_key=missing` and `status=accepted`. Duplicate refunds were observed for three enterprise accounts.

*Evidence: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`, `corpus/tickets/TICKET-8729.md`*

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request carried a `legacy_batch_header`, which bypassed the idempotency check. Single-refund API traffic in the same window included idempotency keys and did not produce duplicates, confirming the bypass as the failure point.

*Evidence: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`, `corpus/tickets/TICKET-8729.md`*

## Follow-up

Reject all bulk-refund requests that lack an idempotency key and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. This is the highest-confidence corrective action supported by the final remediation notes and the evidence that every duplicate refund originated from bulk-refund traffic missing idempotency keys.

*Evidence: `corpus/remediation/notes.md`, `corpus/tickets/TICKET-8729.md`*

## Ambiguity

An initial theory in TICKET-8721 attributed the duplicates to a queue worker replaying an old payment event after a timeout. This theory was formed before API gateway logs were available and is not supported as the primary trigger by the final evidence. Whether worker replay contributed as a secondary factor remains unresolved.

*Evidence: `corpus/tickets/TICKET-8721.md`*
