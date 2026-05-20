# Incident Packet - INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys, causing duplicate refunds for three enterprise accounts. The bulk-refund runbook started at 10:15 UTC; by 10:17 UTC requests with missing idempotency keys were accepted and processed.

Evidence: corpus/timeline/incident_timeline.md, corpus/logs/api_gateway_2026-05-01.log, corpus/tickets/TICKET-8729.md

## Guardrail

The idempotency-required guardrail was skipped at 10:18 UTC because the request carried a legacy_batch_header, which bypassed the guardrail decision. Single-refund API traffic in the same window included idempotency keys and did not duplicate.

Evidence: corpus/timeline/incident_timeline.md, corpus/logs/api_gateway_2026-05-01.log, corpus/tickets/TICKET-8729.md

## Follow-up

Reject all bulk-refund requests that lack an idempotency key, and remove the legacy_batch_header bypass before re-enabling the bulk-refund endpoint. The endpoint was disabled at 10:38 UTC.

Evidence: corpus/remediation/notes.md, corpus/timeline/incident_timeline.md

## Ambiguity

An early theory in TICKET-8721 that a queue worker replayed an old payment event after a timeout is not supported by the final evidence. All confirmed duplicates are tied to bulk-refund requests missing idempotency keys; single-refund traffic was unaffected. The worker-replay theory should be deprioritized but remains unresolved pending further investigation.

Evidence: corpus/tickets/TICKET-8721.md, corpus/tickets/TICKET-8729.md
