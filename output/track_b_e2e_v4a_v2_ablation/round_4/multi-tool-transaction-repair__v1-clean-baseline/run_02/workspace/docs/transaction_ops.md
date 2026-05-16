# Transaction Operations

## Overview

The checkout workflow reserves an order, charges billing, and emits notifications.

## Atomicity Guarantees

- Orders are reserved before billing attempts
- If billing fails, the order is automatically cancelled
- Successful billing marks the order as paid and emits a notification

## Retry Behavior

Retry is safe: failure notifications are emitted exactly once per failed attempt,
avoiding duplicate side effects.
