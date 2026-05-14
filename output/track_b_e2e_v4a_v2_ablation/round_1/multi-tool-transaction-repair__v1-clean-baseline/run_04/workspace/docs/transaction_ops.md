# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification. Retry behavior is described as safe, but the current implementation can emit duplicate failure notifications on retry.

## Atomicity and Retry Behavior

The checkout workflow preserves atomicity by:
1. Reserving an order first
2. Attempting to charge via the billing ledger
3. On billing failure, canceling the order and emitting a single payment_failed notification (when retry=True)
4. On success, marking the order as paid and emitting an order_paid notification

Duplicate failure notifications on retry have been fixed: only one payment_failed notification is emitted per failed checkout attempt.
