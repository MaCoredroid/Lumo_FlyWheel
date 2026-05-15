# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification. Retry behavior is safe: on billing failure, the order is cancelled and a single `payment_failed` notification is emitted (only when retry is enabled).
