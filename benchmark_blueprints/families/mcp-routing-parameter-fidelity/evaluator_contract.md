# `mcp-routing-parameter-fidelity` Evaluator Contract

## Evaluation Goal
Score whether a solver can repair MCP routing behavior without hardcoding only the visible happy path.

## Visible Checks
- `pytest -q tests/test_router.py tests/test_registry.py`

## Trusted Checks
- Routing table remains readable and bounded.
- Provenance logging matches real dispatch decisions.

## Hidden Checks
- Withheld route-selection cases.
- Parameter normalization and coercion fidelity.
- Ambiguity handling without wrong-tool fallback.

## 100-Point Breakdown
- `10`: visible tests pass.
- `25`: route-selection correctness.
- `20`: parameter fidelity.
- `20`: hidden ambiguity handling.
- `15`: provenance logging correctness.
- `10`: docs and bounded diff quality.

## Caps
- If dispatch is hardcoded only for the visible case, total score is capped at `20`.
- If provenance logs stay stale, total score is capped at `25`.
