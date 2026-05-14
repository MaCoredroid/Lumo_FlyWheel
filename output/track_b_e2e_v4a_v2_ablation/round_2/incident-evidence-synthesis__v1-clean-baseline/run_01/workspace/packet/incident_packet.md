# Incident Packet: INC-2047

## Summary

Duplicate refunds were observed for enterprise accounts during a bulk-refund runbook execution.

## Timeline

| Time (UTC) | Event |
|------------|
| 10:15 | Support bulk-refund runbook started for enterprise credits |
| 10:17 | Bulk refund requests accepted without idempotency keys |
| 10:18 | Idempotency-required guardrail skipped due to `legacy_batch_header` |
| 10:25 | Duplicate refunds reported |
| 10:38 | Bulk-refund endpoint disabled |

## Evidence

### Triggering Condition
Bulk-refund requests were accepted without idempotency keys.

### Failed Guardrail
The idempotency-required guardrail was skipped due to the `legacy_batch_header` bypass.

### Supporting Evidence
- TICKET-8721: Initial theory suggested queue worker replay, but this was opened before API gateway logs were available.
- TICKET-8729: Every duplicate has `request_class=bulk-refund` and no idempotency key. Single-refund API traffic in the same window includes idempotency keys and did not duplicate.

## Highest-Confidence Follow-Up Action

Reject bulk-refund requests unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint.

## Unresolved Ambiguity

Worker replay was an early symptom theory but is not supported as the primary trigger by final evidence.
