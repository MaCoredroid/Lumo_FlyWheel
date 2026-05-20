# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys, allowing duplicate refunds for three enterprise accounts. API gateway logs show requests with `request_class=bulk-refund`, `idempotency_key=missing`, and `status=accepted` at 10:17 UTC. One request was flagged `duplicate_candidate=true`.

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request carried the `legacy_batch_header`, which bypassed the check. Single-refund traffic in the same window included idempotency keys and did not duplicate, confirming the bypass was the root cause.

## Follow-up

Reject all bulk-refund requests that lack an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. Additionally, investigate worker replay and retry backoff as noted in early remediation.

## Ambiguity

An early theory (TICKET-8721) attributed duplicates to a queue worker replaying an old payment event after a timeout. This theory was formed before API gateway logs were available. Final evidence (TICKET-8729, gateway logs) does not support worker replay as the primary trigger. The role of worker replay, if any, remains unresolved.
