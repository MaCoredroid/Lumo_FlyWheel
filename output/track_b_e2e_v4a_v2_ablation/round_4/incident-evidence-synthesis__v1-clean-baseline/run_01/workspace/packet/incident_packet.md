# Incident Packet: INC-2047

## Summary

This packet synthesizes evidence from the frozen corpus regarding incident INC-2047, which involved duplicate refunds for enterprise accounts.

## Triggering Condition

Bulk-refund requests were accepted without idempotency keys at 10:17 UTC during the support bulk-refund runbook execution for enterprise credits.

## Failed Guardrail

The idempotency-required guardrail was skipped at 10:18 UTC due to the presence of `legacy_batch_header`, allowing duplicate refund requests to be processed.

## Evidence Chain

| Time (UTC) | Event |
|------------|-------|
| 10:15 | Support bulk-refund runbook started for enterprise credits |
| 10:17 | Bulk refund requests accepted without idempotency keys |
| 10:18 | Idempotency-required guardrail skipped due to `legacy_batch_header` |
| 10:25 | Duplicate refunds reported |
| 10:38 | Bulk-refund endpoint disabled |

## Ticket Analysis

- **TICKET-8721**: Initial theory suggested queue worker replay caused the duplicates. This theory was formed before API gateway logs were available.
- **TICKET-8729**: Evidence confirmed all duplicates had `request_class=bulk-refund` and no idempotency key. Single-refund API traffic in the same window included idempotency keys and did not duplicate.

## Highest-Confidence Follow-Up Action

Reject bulk-refund requests unless they include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the endpoint.

## Unresolved Ambiguity

The worker replay hypothesis (from TICKET-8721) was an early theory but is not supported as the primary trigger by final evidence. The actual root cause is the guardrail bypass allowing non-idempotent bulk-refund requests.
