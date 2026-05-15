# Incident Packet: INC-2047

## Summary

This packet documents the investigation findings for incident INC-2047 involving duplicate refunds for enterprise accounts.

## Triggering Condition

Bulk-refund requests were accepted without idempotency keys during the enterprise credits runbook, enabling duplicate refund processing.

## Failed Guardrail

The idempotency-required guardrail was skipped due to the `legacy_batch_header` bypass, allowing non-idempotent bulk refund requests to proceed.

## Evidence Chain

| Time (UTC) | Event |
|------------|-------|
| 10:15 | Support bulk-refund runbook started for enterprise credits |
| 10:17 | Bulk refund requests accepted without idempotency keys |
| 10:18 | Idempotency guardrail skipped due to `legacy_batch_header` |
| 10:25 | Duplicate refunds reported |
| 10:38 | Bulk-refund endpoint disabled |

## Findings

- **TICKET-8721**: Initial theory of queue worker replay was opened before API gateway logs were available.
- **TICKET-8729**: Evidence confirms every duplicate has `request_class=bulk-refund` and no idempotency key. Single-refund API traffic in the same window includes idempotency keys and did not duplicate.

## Highest-Confidence Follow-Up Action

Reject bulk-refund requests unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint.

## Unresolved Ambiguity

Worker replay was an early symptom theory but is not supported as the primary trigger by final evidence. The lack of idempotency keys in bulk-refund requests remains the confirmed root cause.
