# Incident Packet: INC-2047

## Summary

Duplicate refunds were observed for enterprise accounts during a bulk-refund runbook execution. The root cause was traced to the idempotency guardrail being bypassed for requests with the `legacy_batch_header`.

## Triggering Condition

Bulk-refund requests were accepted without idempotency keys at 10:17 UTC, immediately following the start of the support bulk-refund runbook at 10:15 UTC.

## Failed Guardrail

The idempotency-required guardrail was skipped at 10:18 UTC due to the presence of the `legacy_batch_header`, allowing duplicate refund requests to be processed.

## Evidence

- **TICKET-8721**: Initial symptom report of duplicate refunds for three enterprise accounts. Early theory suggested worker replay, but this was opened before API gateway logs were available.
- **TICKET-8729**: Confirmed that all duplicate requests had `request_class=bulk-refund` and no idempotency key. Single-refund API traffic in the same window included idempotency keys and did not duplicate.
- **Timeline**: Shows the sequence of events from runbook start through endpoint disablement.

## Highest-Confidence Follow-Up Action

Reject bulk-refund requests that do not include an idempotency key. Remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity

The worker replay theory from TICKET-8721 was an early hypothesis formed before API gateway logs were available. Final evidence points to the guardrail bypass as the primary trigger, not worker replay behavior.
