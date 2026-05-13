# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification. Retry behavior is safe - on billing failure with retry=True, exactly one payment_failed notification is emitted and the order is cancelled to preserve to maintain atomicity.
