# Benchmark Run

## Attempt History
- `attempt_01`: solver overfit to tool capability instead of policy.
- `attempt_02`: hardened evaluator produced the target `20/100` behavior.

## Hardening Applied
- Added zero-score policy-violation cap.
- Added hidden checks for technically possible but forbidden actions.
