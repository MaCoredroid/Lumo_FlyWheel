# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits a notification upon success.

## Atomicity

- Orders

1. Order is reserved in the store.
2. Billing charge is attempted.
3. On success: order marked as paid, `order_paid` notification emitted.
4. On failure: order cancelled. If `retry=True`, a single `payment_failed` notification is emitted.

## Retry Behavior

Retry is safe: failure notifications are emitted at most once per failed transaction attempt.
