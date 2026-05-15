# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits notifications.

## Notification Behavior

- On successful payment: emits a single `order_paid` notification.
- On billing failure with retry: emits a single `payment_failed` notification (no duplicates).

Atomicity is preserved: if billing fails, the order is cancelled and no charge is recorded.
