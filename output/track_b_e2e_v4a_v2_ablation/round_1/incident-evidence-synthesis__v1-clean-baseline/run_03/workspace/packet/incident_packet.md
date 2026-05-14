# Incident Packet: INC-2047

## Summary
Duplicate refunds were issued to enterprise accounts due to a guardrail bypass that allowed bulk-refund requests without idempotency keys.

## Triggering Condition
Bulk-refund requests were accepted without idempotency keys at 10:17 UTC. The `legacy_batch_header` was present in requests, which caused the `idempotency-required` guardrail to be skipped at 10:18 UTC.

## Failed Guardrail
- **Guardrail Name:** `idempotency-required`
- **Failure Mode:** Skipped due to `legacy_batch_header` bypass
- **Impact: Duplicate refunds were issued to three enterprise accounts before the bulk-refund endpoint was disabled at 10:38 UTC.

## Highest-Confidence Follow-Up Action
Remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. All bulk-refund requests must include an idempotency key.

## Unresolved Ambiguity
The initial theory (TICKET-8721) suggested investigating worker replay and retry backoff behavior, but API gateway logs indicate the root cause was the guardrail bypass, not worker replay.
