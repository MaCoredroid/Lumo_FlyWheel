# TICKET-8729

Symptom: duplicate refunds observed.

Evidence: every duplicate has `request_class=bulk-refund` and no idempotency key. Single-refund API traffic in the same window includes idempotency keys and did not duplicate.
