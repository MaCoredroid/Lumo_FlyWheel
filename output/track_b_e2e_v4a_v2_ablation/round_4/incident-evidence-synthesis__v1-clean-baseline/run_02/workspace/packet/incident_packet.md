# Incident Packet: INC-2047

## Summary

Duplicate refunds were issued to enterprise accounts due to a guardrail bypass allowing bulk-refund requests without idempotency keys.

## Triggering Condition

Bulk-refund requests were accepted without idempotency keys at 10:17 UTC. Every duplicate refund case had `request_class=bulk-refund` and no idempotency key present.

## Failed Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC due to the presence of `legacy_batch_header`, which acted as a bypass mechanism.

## Highest-Confidence Follow-Up Action

1. Reject bulk-refund requests that do not include an idempotency key
2. Remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint

## Unresolved Ambiguity

The initial theory (TICKET-8721) that a queue worker replayed an old payment event after a timeout was formed before API gateway logs were available. Evidence from TICKET-8729 indicates the root cause was the missing idempotency key validation, not worker replay.

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 10:15 | Support bulk-refund runbook started for enterprise credits |
| 10:17 | Bulk refund requests accepted without idempotency keys |
| 10:18 | `idempotency-required` guardrail skipped due to `legacy_batch_header` |
| 10:25 | Duplicate refunds reported |
| 10:38 | Bulk-refund endpoint disabled |

## Evidence Sources

- `/workspace/corpus/logs/api_gateway_2026-05-01.log`
- `/workspace/corpus/timeline/incident_timeline.md`
- `/workspace/corpus/tickets/TICKET-8721.md`
- `/workspace/corpus/tickets/TICKET-8729.md`
- `/workspace/corpus/remediation/notes.md`
