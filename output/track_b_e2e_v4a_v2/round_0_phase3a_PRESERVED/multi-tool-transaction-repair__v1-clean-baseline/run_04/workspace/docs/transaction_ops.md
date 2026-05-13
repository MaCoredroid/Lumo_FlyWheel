# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification. Retry behavior is safe: on billing failure with retry enabled, exactly one `payment_failed` notification is emitted, avoiding duplicate side effects.
