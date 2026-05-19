# Transaction Operations

Checkout reserves an order, charges billing, and then emits a notification.
On billing failure the order is cancelled and a ledger refund is recorded to preserve atomicity.
Retry behavior emits at most one `payment_failed` notification, avoiding duplicate side effects.
