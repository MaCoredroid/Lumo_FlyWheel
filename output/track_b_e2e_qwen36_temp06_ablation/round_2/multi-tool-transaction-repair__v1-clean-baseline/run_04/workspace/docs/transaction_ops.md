# Transaction Operations

## Checkout Workflow

`checkout()` reserves an order, charges billing, and emits a notification on success.

### Failure handling

- On billing failure the order is **cancelled** and the exception is re-raised.
- When `retry=True`, a single `payment_failed` notification is emitted (not duplicated).
- The order state is updated **before** the notification is emitted so the transaction result is durable even if the notification side effect fails.

### Atomicity

- `mark_paid` is called before `queue.emit` to ensure the order status is persisted first.
- On failure, `cancel_order` runs before any notification, keeping state and side effects consistent.
