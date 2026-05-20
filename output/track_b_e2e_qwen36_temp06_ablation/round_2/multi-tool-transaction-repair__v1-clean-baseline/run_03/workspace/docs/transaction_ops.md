# Transaction Operations

## Checkout Flow

1. **Reserve** the order in the store.
2. **Charge** the billing ledger.
3. On success: **mark paid** and emit a single `order_paid` notification.
4. On failure: **cancel** the order and re-raise the `BillingError`.

## Retry Behavior

When `retry=True` and billing fails, a single `payment_failed` notification is emitted before re-raising. Duplicate notifications are not emitted — the `if retry:` block calls `queue.emit` exactly once.

## Atomicity Guarantees

- **No partial state on failure**: the order is always cancelled when billing raises.
- **No duplicate side effects**: each failure path emits at most one notification.
- **Idempotent cleanup: the `conftest.py` fixture resets all shared state before and after every test.
