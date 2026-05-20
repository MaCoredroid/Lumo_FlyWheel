# Transaction Operations

Checkout reserves an order, charges billing, and then emits a notification.
On billing failure the order is cancelled and the exception propagates.
When retry is enabled, a single payment_failed notification is emitted before re-raising.

## Atomicity

- Order reservation and billing charge are grouped: on BillingError the order is cancelled so it never reaches paid status.
- Notifications are idempotent with the outcome - only one notification per event is emitted.

## Retry behavior

When retry=True and billing fails, exactly one payment_failed notification is emitted before the exception is re-raised. This avoids duplicate side effects.
