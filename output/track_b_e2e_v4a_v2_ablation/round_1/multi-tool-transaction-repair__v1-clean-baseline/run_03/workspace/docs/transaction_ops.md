# Transaction Operations

## Checkout Workflow

The `checkout` function performs the following steps atomically:

1. **Reserve order**: Creates a reserved order entry with the specified amount.
2. **Charge billing**: Attempts to charge the billing ledger.
3. **On success**: Marks the order as paid and emits an `order_paid` notification.
4. **On failure**: Cancels the order and re-raises the billing error.

## Retry Behavior

When `retry=True` and a billing error occurs:
- The order is cancelled to preserve atomicity.
- A single `payment_failed` notification is emitted (no duplicates).
- The billing error is re-raised to the caller.

This ensures safe retry semantics without duplicate notification side effects.
