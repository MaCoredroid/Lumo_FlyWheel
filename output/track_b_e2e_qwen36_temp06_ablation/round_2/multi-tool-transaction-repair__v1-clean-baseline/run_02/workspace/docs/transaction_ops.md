# Transaction Operations

The checkout workflow reserves an order, charges billing, and emits a notification on success. If billing fails, the order is cancelled and the exception propagates. When `retry=True`, a single `payment_failed` notification is emitted before re-raising — duplicate side effects are prevented.
