# Evaluator Contract

## Evaluation Intent

This family measures whether a solver can repair tool-catalog fallback behavior after a preferred tool id disappears. Simple alias restoration must stay below `30/100`.

## Required Inputs

- Final catalog/router implementation
- Final manifest and router config
- Final operator docs
- Visible and hidden test results

## Scoring Breakdown

- `20 points`: visible suite passes in `workspace/tests`
- `20 points`: preferred-missing browser cases choose the capability- and policy-eligible fallback
- `20 points`: preferred-present cases still select the primary tool when it is eligible
- `20 points`: selection policy is driven from `workspace/tooling/tool_manifest.yaml` and `workspace/config/tool_router.toml` without embedded default routing tables
- `20 points`: docs are current and the solver did not edit or add tests in `workspace/tests`

## Scoring Caps

- Visible checks alone cap the run at `20/100`
- Any embedded default routing table, capability map, or policy list that can satisfy the task without manifest/config data caps the run at `20/100`
- Any added, deleted, or weakened test in `workspace/tests` caps the run at `20/100`
- Any alias-only fix caps the run at `15/100`
- Any blanket-default fallback caps the run at `15/100`

## Visible Checks

- `cd workspace && python -m pytest -q tests`

## Hidden Checks

- Preferred-missing fixtures with multiple fallbacks where only one candidate is capability- and policy-eligible
- Preferred-present fixtures to ensure the solver did not degrade the primary path
- Reject alias-only fixes that never exercise capability filtering
- Reject blanket-default fallback behavior
- Reject stale legacy ids in docs or config

## Fail-Fast Conditions

- First listed fallback always wins
- Preferred tool is ignored even when present and eligible
- Legacy id remains in manifest, config, or docs

## Red-Team Expectations

- A shallow solver will rename the tool id and maybe add an alias
- A fixture-overfitting solver may hardcode one hidden ordering pattern
- Hidden cases should vary fallback order and capability combinations enough to keep both under `30/100`

## Hardness Judgment

- Current target judgment: naive GPT-5.4/high should remain under `30/100`
- Residual risk: narrow hidden capability matrices may still allow heuristic shortcut implementations
