# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification. Retry behavior is safe: on billing failure, the order is cancelled and at most one failure notification is emitted.
