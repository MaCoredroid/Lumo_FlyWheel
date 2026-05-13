# TICKET-8721

Symptom: duplicate refunds observed for three enterprise accounts.

Initial theory: queue worker replayed an old payment event after a timeout. This theory was opened before API gateway logs were available.
