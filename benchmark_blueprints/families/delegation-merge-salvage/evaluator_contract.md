# Evaluator Contract: `delegation-merge-salvage`

## 100-Point Breakdown

- 50 points: deterministic code correctness
- 20 points: artifact discipline across kept/rejected worker hunks
- 10 points: docs and reviewer-note quality
- 10 points: variant-context grounding
- 10 points: verification evidence

## Deterministic correctness

- markdown output contains the watchlist follow-up when requested
- JSON output stays byte-for-byte identical on the baseline fixture
- unrelated fixtures and tests are unchanged

## Score caps

- missing or generic salvage postmortem: max 30
- Worker A wholesale result: max 20
- Worker B wholesale result: max 25
- JSON contract drift: max 20
- lost watchlist follow-up: max 25
- unrelated fixture churn: max 20
