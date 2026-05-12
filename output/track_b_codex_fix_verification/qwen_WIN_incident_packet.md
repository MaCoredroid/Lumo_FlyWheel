# Incident Packet: INC-2047

## Summary

Duplicate refunds were observed for enterprise accounts due to bulk-refund requests being accepted without idempotency keys, triggered by a guardrail bypass.

---

## Triggering Condition

Bulk-refund requests were submitted without idempotency keys. The `idempotency-required` guardrail was skipped due to the presence of `legacy_batch_header`, allowing duplicate requests to be processed.

**Evidence:**
- `corpus/logs/api_gateway_2026-05-01.log`: At 10:17:44Z and 10:17:46Z, bulk-refund requests with `idempotency_key=missing` were accepted.
- `corpus/timeline/incident_timeline.md`: At 10:18 UTC, the idempotency-required guardrail was skipped due to `legacy_batch_header`.

---

## Failed Guardrail

The `idempotency-required` guardrail failed to enforce idempotency key validation because it contained a bypass condition for `legacy_batch_header`.

**Evidence:**
- `corpus/logs/api_gateway_2026-05-01.log`: `guardrail=idempotency-required decision=skipped reason=legacy_batch_header`
- `corpus/tickets/TICKET-8729.md`: All duplicates have `request_class=bulk-refund` and no idempotency key; single-refund traffic with idempotency keys did not duplicate.

---

## Highest-Confidence Follow-Up Action

Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the endpoint.

**Evidence:**
- `corpus/remediation/notes.md`: "bulk-refund requests must be rejected unless they include an idempotency key; remove the `legacy_batch_header` bypass before re-enabling the endpoint."

---

## Unresolved Ambiguity

The initial theory that a queue worker replayed an old payment event after a timeout was opened before API gateway logs were available and has not been confirmed or ruled out.

**Evidence:**
- `corpus/tickets/TICKET-8721.md`: "Initial theory: queue worker replayed an old payment event after a timeout. This theory was opened before API gateway logs were available."
- `corpus/remediation/notes.md`: Early note: investigate worker replay and retry backoff."
