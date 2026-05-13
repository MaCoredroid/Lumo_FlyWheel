# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification.

## Atomicity and Retry Behavior

The checkout workflow maintains atomicity by:
- Reserving an order first
- Attempting to charge via the billing ledger
- On billing failure: cancelling the order and emitting a single `payment_failed` notification
- On success: marking the order as paid and emitting `order_paid`

Retry behavior is now safe: when `retry=True` is passed, only one `payment_failed` notification is emitted, preventing duplicate side effects.
