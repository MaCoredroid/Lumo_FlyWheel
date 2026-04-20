# `rfc-spec-retrieval-grounding` Evaluator Contract

## Evaluation Goal
Score whether a solver can answer a protocol or API question from a frozen corpus with correct, pinned evidence.

## Visible Checks
- `pytest -q tests/test_claim_alignment.py`

## Trusted Checks
- Final answer resolves the benchmark-owned question correctly.
- Citations use the required pinned format.
- No unsupported claim is stated as normative fact.

## Hidden Checks
- Support-document recall and precision on withheld corpus slices.
- Claim-level citation grounding against gold evidence spans.
- Hard-negative rejection for stale drafts and misleading summaries.

## 100-Point Breakdown
- `10`: visible citation-shape checks pass.
- `25`: final answer correctness.
- `25`: support-document recall and precision.
- `20`: claim-level citation grounding.
- `20`: synthesis completeness and contradiction handling.

## Caps
- If citations are missing, total score is capped at `20`.
- If the answer is correct but evidence is unsupported, total score is capped at `25`.
- If hard negatives are used as primary support, total score is capped at `15`.

## Passing Interpretation
- Target naive score: `20/100`.
- Scores above `30` indicate the family is still too shallow.
