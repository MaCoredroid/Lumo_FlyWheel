# Incident Packet: INC-2047

## Triggering Condition
Bulk-refund requests were accepted without idempotency keys at 10:17 UTC, leading to duplicate refunds for enterprise accounts.

## Failed Guardrail
The `idempotency-required` guardrail was skipped at 10:18 UTC due to the `legacy_batch_header` bypass, allowing duplicate refund requests to proceed.

## Highest-Confidence Follow-Up Action
Reject all bulk-refund requests that do not include an idempotency key, and remove the `legacy_batch_header` bypass before re-enabling the bulk-refund endpoint.

## Unresolved Ambiguity
The initial theory in TICKET-8721 suggested a queue worker replayed an old payment event after a timeout, but API gateway logs (available after the ticket was opened) indicate the root cause was the guardrail bypass, not worker replay.
