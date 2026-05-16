# Transaction Operations

## Checkout Workflow

The `checkout` function performs the following steps atomically:

1. Reserves an order in the order store
2. Charges the billing ledger
3. If billing fails:
   - Cancels the reserved order
   - Emits a single `payment_failed` notification (only when `retry=True`)
   - Re-raises the billing error
4. If billing succeeds:
   - Marks the order as paid
   - Emits an `order_paid` notification

## Atomicity Guarantees

- Orders are always left in a consistent state (reserved, paid, or cancelled)
- Failed billing always results in order cancellation
- No duplicate notifications are emitted on retry failures

## Notification Side Effects

- `order_paid`: Emitted once when payment succeeds
- `payment_failed`: Emitted at most once when payment fails with `retry=True`
