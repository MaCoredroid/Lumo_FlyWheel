# Transaction Operations

Current note: checkout reserves an order, charges billing, and then emits a notification. Retry behavior is described as safe, but the current implementation can emit duplicate failure notifications on retry.
