# Incident Packet: INC-2047

## Summary

Duplicate Refunds in Bulk-Refund Endpoint

**Date:** 2026-05-01  
**Status:** Resolved (endpoint disabled)

---

## Triggering Condition

Bulk refund requests were accepted without idempotency keys. At 10:17 UTC, bulk refund requests for enterprise credits were processed without the required idempotency keys, leading to duplicate refund transactions.

---

## Failed Guardrail

The `idempotency-required` guardrail was skipped at 10:18 UTC due to the presence of a `legacy_batch_header`. This bypass allowed duplicate requests to proceed unchecked.

**Log Evidence:**
```
2026-05-01T10:18:02Z guardrail=idempotency-required decision=skipped reason=legacy_batch_header
```

---

## Highest-Confidence Follow-Up Action

1. **Reject bulk-refund requests** that do not include an idempotency key
2. **Remove the `legacy_batch_header` bypass** before re-enabling the bulk-refund endpoint

This action is directly supported by the final remediation note and correlates with evidence showing that single-refund API traffic (which includes idempotency keys) did not experience duplicates.

---

## Unresolved Ambiguity

The initial theory that a queue worker replayed an old payment event after a timeout was opened before API gateway logs were available. While evidence now points to the missing idempotency keys as the root cause, the original worker replay hypothesis was not explicitly ruled out in the available documentation.

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 10:15 | Support bulk-refund runbook started for enterprise credits |
| 10:17 | Bulk refund requests accepted without idempotency keys |
| 10:18 | Idempotency-required guardrail skipped due to `legacy_batch_header` |
| 10:25 | Duplicate refunds reported |
| 10:38 | Bulk-refund endpoint disabled |

---

## Related Tickets
- TICKET-8721: Initial report of duplicate refunds for three enterprise accounts
- TICKET-8729: Evidence linking duplicates to missing idempotency keys
