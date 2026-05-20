# Transaction Operations

## Checkout Flow

`checkout(order_id, amount, *, fail_billing=False, retry=False)` performs the following steps atomically:

1. **Reserve** the order via `store.reserve_order`.
2. **Charge** billing via `ledger.charge`.
3. On success: **mark paid** and emit a single `order_paid` notification.
4. On billing failure: **cancel** the order and re-raise `BillingError`.
   - If `retry=True`, a single `payment_failed` notification is emitted before re-raising.

## Atomicity Guarantees

- **No duplicate notifications**: the `payment_failed` event is emitted at most once per failed checkout, even when `retry=True`.
- **Order state consistency**: a failed charge always results in `cancelled` status; the order is never left in `reserved` state.
- **Ledger integrity**: a `charge` entry is only appended when billing succeeds; no phantom charges are recorded on failure.
