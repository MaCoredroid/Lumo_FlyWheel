# Transaction Operations

`checkout` reserves an order, charges billing, and then emits a notification.
On billing failure the order is cancelled and a single `payment_failed` notification is emitted when `retry=True`. Retry behavior is safe — failure notifications are never duplicated.
