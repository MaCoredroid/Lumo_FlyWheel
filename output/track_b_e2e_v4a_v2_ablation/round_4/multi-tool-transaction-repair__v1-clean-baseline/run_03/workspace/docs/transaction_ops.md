# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits a notification.

## Atomicity

The workflow ensures atomicity by:
1. Reserving the order first
2. Attempting the billing charge
3. On failure: cancelling the order and optionally emitting a single payment_failed notification (when retry=True)
4. On success: marking the order as paid and emitting an order_paid notification

## Retry Behavior

When `retry=True` and billing fails, exactly one `payment_failed` notification is emitted. Duplicate notifications are prevented by design.
