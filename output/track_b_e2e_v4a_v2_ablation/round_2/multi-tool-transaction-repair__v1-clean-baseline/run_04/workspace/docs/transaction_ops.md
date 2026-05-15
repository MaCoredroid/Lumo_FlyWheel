# Transaction Operations

Checkout reserves an order, charges billing, and then emits a notification. On billing failure, the order is cancelled and a single `payment_failed` notification is emitted when retry is enabled. This ensures atomicity without duplicate notification side effects.
