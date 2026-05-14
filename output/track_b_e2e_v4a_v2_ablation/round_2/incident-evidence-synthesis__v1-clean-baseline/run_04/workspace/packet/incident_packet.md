# Incident Packet: INC-2047

## Summary

## Triggering Condition

Bulk refund requests were accepted without idempotency keys at 10:17 UTC, leading to duplicate refunds for three enterprise accounts reported at 10:25 UTC.

## Failed Guardrail

The idempotency-required guardrail was bypassed at 10:18 UTC due to the `legacy_batch_header` being present in the request, allowing duplicate refund processing.

## Highest-Confidence Follow-Up Action

Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity

The initial theory (TICKET-8721) that queue worker replay caused the duplicates was speculative and opened before API gateway logs were available. While TICKET-8729 provides strong evidence linking duplicates to missing idempotency keys in bulk-refund requests, the worker replay hypothesis was not fully ruled out.
