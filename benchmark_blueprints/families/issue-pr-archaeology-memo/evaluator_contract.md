# `issue-pr-archaeology-memo` Evaluator Contract

## Evaluation Goal
Score whether a solver can reconstruct causality from a frozen issue or PR archive rather than echoing the loudest thread.

## Visible Checks
- `pytest -q tests/test_archaeology_memo.py`

## Trusted Checks
- Memo names the correct originating change.
- Recommendation scope matches the evidence.
- Timeline ordering is internally consistent.

## Hidden Checks
- Gold causal timeline alignment.
- Rejection of downstream suspects.
- Evidence anchoring to issues, PRs, and commit excerpts.

## 100-Point Breakdown
- `10`: visible memo-shape checks pass.
- `25`: root-cause correctness.
- `25`: causal timeline quality.
- `20`: rejection of downstream suspects.
- `20`: evidence grounding and remediation fit.

## Caps
- If the memo names a downstream symptom as the root cause, total score is capped at `20`.
- If the memo omits explicit evidence, total score is capped at `25`.

## Passing Interpretation
- Target naive score: `20/100`.
