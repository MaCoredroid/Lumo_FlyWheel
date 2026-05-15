# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits a notification on success.

## Atomicity

- If billing fails, the order is automatically cancelled and no charge is recorded.
- On successful billing, the order is marked as paid and an `order_paid` notification is emitted.

## Retry Behavior

When a billing failure occurs with `retry=True`, a single `payment_failed` notification is emitted. The implementation ensures no duplicate failure notifications are sent.
