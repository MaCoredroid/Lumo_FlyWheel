# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys during the enterprise-credits bulk-refund runbook. At 10:17 UTC the API gateway accepted `request_class=bulk-refund` requests with `idempotency_key=missing` and `retry_after_ms=0`, allowing duplicate refunds to be processed ([`corpus/logs/api_gateway_2026-05-01.log`](corpus/logs/api_gateway_2026-05-01.log)).

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because of the presence of `legacy_batch_header`, which bypassed the idempotency check ([`corpus/logs/api_gateway_2026-05-01.log`](corpus/logs/api_gateway_2026-05-01.log)). This is consistent with the timeline: bulk-refund requests were accepted without idempotency keys (10:17 UTC), then the guardrail skip was logged (10:18 UTC) ([`corpus/timeline/incident_timeline.md`](corpus/timeline/incident_timeline.md)).

## Follow-Up

Per remediation notes, bulk-refund requests must be rejected unless they include an idempotency key, and the `legacy_batch_header` bypass must be removed before the endpoint is re-enabled ([`corpus/remediation/notes.md`](corpus/remediation/notes.md)).

## Ambiguity

An early theory (TICKET-8721) attributed the incident to a queue worker replaying an old payment event after a timeout. However, final evidence does not support this as the primary trigger: every duplicate refund shares `request_class=bulk-refund` with no idempotency key, while single-refund traffic in the same window included idempotency keys and did not duplicate ([`corpus/tickets/TICKET-8721.md`](corpus/tickets/TICKET-8721.md), [`corpus/tickets/TICKET-8729.md`](corpus/tickets/TICKET-8729.md)). Worker replay and retry backoff remain unresolved.

