# Benchmark Run

## Attempt History
- `attempt_01`: happy-path repair overperformed before rollback checks were weighted correctly.
- `attempt_02`: hardened evaluator held the family near `20/100`.

## Hardening Applied
- Added hidden rollback-state checks.
- Added duplicate-side-effect penalties.
- Reduced credit for operator-note-only fixes.
