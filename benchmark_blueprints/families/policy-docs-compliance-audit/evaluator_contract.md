# `policy-docs-compliance-audit` Evaluator Contract

## Evaluation Goal
Score whether a solver can classify compliance findings from a frozen policy corpus without inventing authority.

## Visible Checks
- `pytest -q tests/test_compliance_audit.py`

## Trusted Checks
- Findings use the required schema.
- Classification matches the gold policy outcome.

## Hidden Checks
- Authority ordering between policy, runbook, and implementation notes.
- Correct handling of narrow compliant exceptions.
- Rejection of non-authoritative policy-like language.

## 100-Point Breakdown
- `10`: visible schema checks pass.
- `25`: finding correctness.
- `25`: authority ranking correctness.
- `20`: compliant-exception handling.
- `20`: documentation-drift versus policy-violation separation.
