# Incident Packet: INC-2047

## Trigger

The triggering condition was bulk-refund traffic being accepted without idempotency keys. At 10:17 UTC, the timeline records bulk refund requests accepted without idempotency keys, and the API gateway log shows two accepted `request_class=bulk-refund` requests with `idempotency_key=missing`, including one marked `duplicate_candidate=true`.

Evidence:
- `corpus/timeline/incident_timeline.md`
- `corpus/logs/api_gateway_2026-05-01.log`
- `corpus/tickets/TICKET-8729.md`

## Guardrail

The failed guardrail was `idempotency-required`. It was skipped at 10:18 UTC because of `legacy_batch_header`, allowing bulk-refund requests without idempotency keys to proceed.

Evidence:
- `corpus/timeline/incident_timeline.md`
- `corpus/logs/api_gateway_2026-05-01.log`

## Follow-up

The highest-confidence follow-up is to reject bulk-refund requests unless they include an idempotency key and remove the `legacy_batch_header` bypass before re-enabling the endpoint. This matches the final remediation note and directly addresses the accepted missing-key requests and skipped guardrail.

Evidence:
- `corpus/remediation/notes.md`
- `corpus/logs/api_gateway_2026-05-01.log`

## Ambiguity

Worker replay and retry backoff remain an early investigative theory, but the final evidence does not support them as the primary trigger. TICKET-8721 opened the worker-replay theory before API gateway logs were available; later evidence ties every duplicate to bulk-refund traffic without idempotency keys, while single-refund traffic with keys did not duplicate.

Evidence:
- `corpus/tickets/TICKET-8721.md`
- `corpus/tickets/TICKET-8729.md`
- `corpus/remediation/notes.md`
