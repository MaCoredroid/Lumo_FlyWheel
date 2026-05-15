# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits notifications.

## Atomicity is preserved: on billing failure, the order is cancelled and no "order_paid" notification is emitted.

## Retry Behavior

When `retry=True` and billing fails, a single "payment_failed" notification is emitted (no duplicates).
