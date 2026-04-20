# `dependency-upgrade-research-pack` Evaluator Contract

## Evaluation Goal
Score whether a solver can assemble an upgrade decision from a frozen evidence pack instead of echoing one side of the tradeoff.

## Visible Checks
- `pytest -q tests/test_upgrade_pack.py`

## Trusted Checks
- Recommendation matches the gold decision.
- Risk matrix is complete and evidence-backed.

## Hidden Checks
- External changelog and migration evidence are handled correctly.
- Internal call-site search is incorporated.
- Stale reverted risks are rejected.

## 100-Point Breakdown
- `10`: visible pack-shape checks pass.
- `25`: recommendation correctness.
- `25`: risk evidence quality.
- `20`: internal/external synthesis.
- `20`: false-positive rejection and staged-plan fit.

## Caps
- If the answer uses only external docs or only internal search, total score is capped at `25`.
- If stale reverted risks are treated as active blockers, total score is capped at `15`.
