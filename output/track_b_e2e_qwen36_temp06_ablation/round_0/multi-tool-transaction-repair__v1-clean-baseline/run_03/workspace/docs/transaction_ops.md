# Transaction Operations

Checkout reserves an order, charges billing, and then emits a notification.
On billing failure the order is cancelled (atomic rollback) and, when retry is
enabled, a single `payment_failed` notification is emitted — no duplicates.
