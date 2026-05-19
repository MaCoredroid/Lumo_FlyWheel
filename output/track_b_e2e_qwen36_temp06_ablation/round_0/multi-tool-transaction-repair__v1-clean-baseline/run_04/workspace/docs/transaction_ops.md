# Transaction Operations

checkout reserves an order, charges billing, and then emits a notification.
On billing failure the order is cancelled and a `payment_failed` notification is emitted exactly once.
Retry behavior is safe — no duplicate side effects are produced.
