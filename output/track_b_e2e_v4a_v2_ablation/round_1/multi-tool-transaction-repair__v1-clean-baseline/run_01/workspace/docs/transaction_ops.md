# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification.

## Atomicity Guarantees

- Orders are reserved before billing is attempted.
- If billing fails, the order is automatically cancelled.
- On billing failure with retry enabled, a single `payment_failed` notification is emitted.
- Duplicate failure notifications are prevented by emitting only once per failure.
