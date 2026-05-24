# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys during the enterprise-credits bulk-refund runbook. At 10:17 UTC the API gateway accepted `request_class=bulk-refund` requests with `idempotency_key=missing` and `retry_after_ms=0`, leading to duplicate refunds. Single-refund API traffic in the same window included idempotency keys and did not duplicate.

Sources: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`, `corpus/tickets/TICKET-8729.md`

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because of the `legacy_batch_header` bypass. This allowed duplicate bulk-refund requests to be processed without deduplication.

Sources: `corpus/logs/api_gateway_2026-05-01.log`, `corpus/timeline/incident_timeline.md`

## Follow-Up

Bulk-refund requests must be rejected unless they include an idempotency key. The `legacy_batch_header` bypass must be removed before the endpoint is re-enabled.

Source: `corpus/remediation/notes.md`

## Ambiguity

An early theory (TICKET-8721) that the incident was caused by queue worker replay is not supported by final evidence. Worker replay and retry backoff remain unresolved.

Sources: `corpus/tickets/TICKET-8721.md`, `corpus/tickets/TICKET-8729.md`
