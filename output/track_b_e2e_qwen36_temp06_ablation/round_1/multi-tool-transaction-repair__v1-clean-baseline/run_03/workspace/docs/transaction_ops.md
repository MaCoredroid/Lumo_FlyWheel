# Transaction Operations

## Checkout Flow

1. **Reserve** the order in the order store.
2. **Charge** the billing gateway.
3. On **success**: mark the order as paid and emit a single `order_paid` notification.
4. On **failure**: cancel the order. If `retry=True`, emit exactly one `payment_failed` notification. Re-raise the billing error.

## Atomicity

The checkout workflow preserves atomicity: billing charge and order-state changes are paired — a successful charge always results in `mark_paid`, and a failed charge always results in `cancel_order`. Notifications are emitted at most once per event, so no duplicate side effects occur.
