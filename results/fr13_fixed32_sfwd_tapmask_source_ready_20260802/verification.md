# Verification

- Fixed-parent topology derivation: all 32 physical rows passed.
- Exact prior masks: tap 0 rows 0-8, tap 1 rows 0-3, tap 2 row 0.
- Exact prior-state values within each mask: passed.
- Focused descriptorless launcher and kernel tests: 24 passed.
- Ruff, Python byte compilation, and diff checks: passed.
- Candidate identity is distinct from the fixed-stride parent.
- Codegen, runtime byte correctness, timing, and floor acceptance: not run.
