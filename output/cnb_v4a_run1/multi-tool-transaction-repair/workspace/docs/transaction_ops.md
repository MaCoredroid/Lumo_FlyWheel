# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits a notification on success. If billing fails, the order is cancelled and the exception is re-raised to preserve atomicity.

## Retry Behavior

When `retry=True` and billing fails, a single `payment_failed` notification is emitted. Duplicate side effects are avoided by ensuring `queue.emit` is called at most once per failure.
