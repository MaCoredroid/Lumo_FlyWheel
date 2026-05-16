# Transaction Operations

The `checkout` function reserves an order, charges billing, and emits a notification.

## Atomicity Guarantees

- Order reservation and billing charge are handled atomically.
- On billing failure, the order is automatically cancelled.
- Retry behavior is safe: failure notifications are emitted exactly once per failure, avoiding duplicate side effects.
