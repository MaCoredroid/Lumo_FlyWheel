# Incident Packet — INC-2047

## Trigger

Bulk-refund requests were accepted without idempotency keys, allowing duplicate refunds to be processed.

- API gateway logs at 10:17 UTC show `request_class=bulk-refund` with `idempotency_key=missing` and `status=accepted`.
- Timeline confirms bulk refund requests accepted without idempotency keys at 10:17 UTC.
- TICKET-8729: every duplicate refund has `request_class=bulk-refund` and no idempotency key; single-refund traffic in the same window included keys and did not duplicate.

## Guardrail

The `idempotency-required` guardrail was skipped because of the `legacy_batch_header` bypass.

- API gateway log at 10:18 UTC: `guardrail=idempotency-required decision=skipped reason=legacy_batch_header`.
- Timeline: idempotency-required guardrail skipped due to `legacy_batch_header` at 10:18 UTC.

## Follow-up

Reject bulk-refund requests that lack an idempotency key and remove the `legacy_batch_header` bypass before re-enabling the endpoint.

- Remediation notes: "bulk-refund requests must be rejected unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint."

## Ambiguity

The initial theory (TICKET-8721) attributed the duplicates to a queue worker replaying an old payment event after a timeout. Final evidence (TICKET-8729) does not support this as the primary trigger—all duplicates correlate with bulk-refund requests missing idempotency keys, while single-refund traffic was unaffected. The worker replay hypothesis remains unresolved but is not the leading explanation.
