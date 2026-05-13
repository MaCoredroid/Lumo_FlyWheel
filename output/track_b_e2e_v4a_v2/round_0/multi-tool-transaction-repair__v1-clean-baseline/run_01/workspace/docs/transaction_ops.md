# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits a notification on success.

## Atomicity

- Orders are reserved before billing is attempted.
- If billing fails, the order is cancelled to maintain atomicity.
- On retry with billing failure, a single "payment_failed" notification is emitted (no duplicates).
- On successful payment, an "order_paid" notification is emitted.
