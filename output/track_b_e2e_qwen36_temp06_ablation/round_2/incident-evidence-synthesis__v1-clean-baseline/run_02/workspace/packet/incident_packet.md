# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys on 2026-05-01 at 10:17 UTC. The bulk-refund runbook for enterprise credits started at 10:15 UTC; requests were processed with `request_class=bulk-refund`, `idempotency_key=missing`, and `status=accepted`. Duplicate refunds were subsequently reported for three enterprise accounts.

## Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC because the request carried a `legacy_batch_header`, which bypassed the idempotency check. API gateway logs confirm `guardrail=idempotency-required decision=skipped reason=legacy_batch_header`. Single-refund API traffic in the same window included idempotency keys and did not duplicate, confirming the guardrail skip as the differentiating factor.

## Follow-Up

Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. This aligns with the final remediation note and is the highest-confidence corrective action supported by the evidence.

## Ambiguity

An early theory (TICKET-8721) attributed the duplicates to a queue worker replaying an old payment event after a timeout. This theory was formed before API gateway logs were available and is not supported as the primary trigger by the final evidence (TICKET-8729 and gateway logs). Whether worker replay contributed to the incident remains unresolved.
