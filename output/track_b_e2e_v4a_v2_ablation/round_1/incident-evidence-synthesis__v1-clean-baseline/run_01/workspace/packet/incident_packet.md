# Incident Packet: INC-2047

## Summary

This packet synthesizes evidence from the frozen corpus regarding duplicate refunds observed for enterprise accounts.

## Triggering Condition

Bulk-refund requests were accepted without idempotency keys at 10:17 UTC.

## Failed Guardrail

The idempotency-required guardrail was skipped due to the presence of `legacy_batch_header` at 10:18 UTC.

## Evidence from TICKET-8729 confirms every duplicate refund has `request_class=bulk-refund` and no idempotency key, while single-refund API traffic in the same window included idempotency keys and did not duplicate.

## Highest-Confidence Follow-Up Action

Reject bulk-refund requests unless they include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the endpoint.

## Unresolved Ambiguity

TICKET-8721 initially theorized that a queue worker replayed an old payment event after a timeout. However, this theory was opened before API gateway logs were available, and final evidence (TICKET-8729) does not support worker replay as the primary trigger. The worker replay theory remains an unresolved ambiguity.

## Timeline

- 10:15 UTC: Support bulk-refund runbook started for enterprise credits.
- 10:17 UTC: Bulk refund requests accepted without idempotency keys.
- 10:18 UTC: Idempotency-required guardrail skipped due to `legacy_batch_header`.
- 10:25 UTC: Duplicate refunds reported.
- 10:38 UTC: Bulk-refund endpoint disabled.
