# Incident Packet: INC-2047

## Summary

Duplicate refunds were observed for enterprise accounts during a bulk-refund operationally due to idempotency guardrail bypass.

## Triggering Condition

Bulk-refund requests were accepted without idempotency keys at 10:17 UTC.

## Failed Guardrail

The idempotency-required guardrail was skipped due to the `legacy_batch_header` being present (10:18 UTC).

## Evidence

- **TICKET-8729**: Every duplicate request had `request_class=bulk-refund` and no idempotency key. Single-refund API traffic in the same window included idempotency keys and did not duplicate.
- **Timeline**: At 10:18 UTC, the idempotency guardrail was skipped specifically because of `legacy_batch_header`.
- **TICKET-8721**: Initial theory of queue worker replay was opened before API gateway logs were available.

## Highest-Confidence Follow-Up Action

Reject bulk-refund requests unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint.

## Unresolved Ambiguity

Worker replay was proposed as an early theory (TICKET-8721) but is not supported by final evidence. The primary trigger is the guardrail bypass, not worker replay.
