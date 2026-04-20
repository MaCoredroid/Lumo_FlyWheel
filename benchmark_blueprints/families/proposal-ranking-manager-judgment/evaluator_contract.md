# `proposal-ranking-manager-judgment` Evaluator Contract

## Evaluation Goal
Score whether a solver can make a good managerial choice among several plausible implementation proposals.

## Visible Checks
- `pytest -q tests/test_ranking_brief.py`

## Trusted Checks
- Selected proposal matches the gold choice.
- Brief is grounded in repo evidence and constraints.

## Hidden Checks
- Full ranking order.
- Correct treatment of staffing and rollout tradeoffs.
- Accurate rejection reasoning for losing proposals.

## 100-Point Breakdown
- `10`: visible brief-shape checks pass.
- `20`: proposal ranking correctness.
- `20`: decision quality.
- `20`: constraint integration.
- `15`: rejection reasoning.
- `15`: evidence-backed brief quality.
