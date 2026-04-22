# Integration report

## Backend
- Updated the active approval state to `human_review_required`.
- Left `manual_review` only as a narrow legacy compatibility mapping.

## Frontend
- Updated the seeded form state and request client to submit `human_review_required`.

## Docs
- Rewrote the runbook so operators verify the new token in request and echo order.

## Proof
- Captured the seeded request payload and server echo for `rel-ship-0422`.
- The report keeps request before echo because release fanout evaluates the request path first.
- INC-204 is called out explicitly; no dual-write fallback remains in the live path.
