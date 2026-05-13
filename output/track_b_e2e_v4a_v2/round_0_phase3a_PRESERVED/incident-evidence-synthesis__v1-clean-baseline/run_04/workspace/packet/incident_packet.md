# Incident Packet: INC-2047

## Summary

Duplicate refunds were issued for enterprise accounts.

## Triggering Condition

Bulk refund requests were accepted without idempotency keys at 10:17 UTC on 2026-05-01. The `idempotency-required` guardrail was skipped due to the presence of a `legacy_batch_header`, allowing duplicate requests to be processed.

## Failed Guardrail

**Guardrail:** `idempotency-required`

**Failure:** The guardrail decision was `skipped` because of the `legacy_batch_header` bypass. This allowed bulk-refund requests lacking idempotency keys to be accepted, resulting in duplicate refunds for enterprise accounts.

**Evidence:**
- API Gateway log at 10:18:02Z shows: `guardrail=idempotency-required decision=skipped reason=legacy_batch_header`
- Requests at 10:17:44Z and 10:17:46Z show `idempotency_key=missing status=accepted`

## Highest-Confidence Follow-Up Action

Remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint. All bulk-refund requests must include an idempotency key to prevent duplicate processing.

**Rationale:** The remediation notes explicitly state: "bulk-refund requests must be rejected unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint."

## Unresolved Ambiguity

**Initial theory vs. evidence:** TICKET-8721 initially theorized that a queue worker replayed an old payment event after a timeout. However, TICKET-8729's evidence shows that duplicates only occurred in `bulk-refund` requests without idempotency keys, while single-refund API traffic (which includes idempotency keys) did not duplicate. The worker replay theory was opened before API gateway logs were available and is not supported by the actual evidence.

**Outstanding question:** Whether the initial queue worker timeout theory has any validity, or if the duplicate refunds were solely caused by the guardrail bypass.
