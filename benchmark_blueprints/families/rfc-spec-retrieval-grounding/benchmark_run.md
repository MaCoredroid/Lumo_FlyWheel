# Benchmark Run

## Attempt History
- `attempt_01`
  - model: `gpt-5.4`
  - reasoning: `high`
  - result: over target when answer correctness was over-weighted relative to grounding.
- `attempt_02`
  - model: `gpt-5.4`
  - reasoning: `high`
  - result: `20/100` after claim-level citation and hard-negative checks were added.

## Hardening Applied
- Added support-document precision checks.
- Added hard-negative rejection for stale drafts.
- Moved most score behind grounded citation behavior, not answer text alone.

## Final Judgment
- Target achieved: `yes`
- Current hardness judgment: a naive solver can produce a plausible answer, but without precise grounding it stays near `20/100`.
