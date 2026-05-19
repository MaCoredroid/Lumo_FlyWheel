# Transaction Operations

Checkout reserves an order, charges billing, and then emits a notification. On billing failure, the order is cancelled and a single `payment_failed` notification is emitted when retry is enabled. Retry behavior is now safe and does not produce duplicate side effects.
