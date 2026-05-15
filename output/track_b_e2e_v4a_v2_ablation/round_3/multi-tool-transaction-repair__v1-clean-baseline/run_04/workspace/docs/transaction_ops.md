# Transaction Operations

The `checkout` workflow performs the following steps:

1. Reserves an order via `store.reserve_order()`
2. Attempts to charge via `ledger.charge()`
3. On success: marks order as paid and emits `order_paid` notification
4. On failure: cancels the order and (if `retry=True`) emits a single `payment_failed` notification

The workflow preserves atomicity: if billing fails, the order is cancelled and no charge is recorded.
Duplicate notifications are prevented by emitting `payment_failed` exactly once when retry is enabled.
