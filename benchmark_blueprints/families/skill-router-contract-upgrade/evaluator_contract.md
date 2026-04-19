# Evaluator Contract

## Evaluation Intent

This family measures whether a solver can upgrade a skill router from a single-trigger contract to a multi-field routing contract with suppressors, input gates, and disciplined fallback behavior. Passing visible tests alone must not yield more than `20/100`.

## Required Inputs

- Final router implementation
- Final routing config and docs examples
- Final routing fixtures
- Visible and hidden test results

## Scoring Breakdown

- `20 points`: visible suite passes in `workspace/tests`
- `20 points`: list triggers and required-input gating work across additional unseen prompt shapes
- `20 points`: negative triggers suppress false positives in ambiguous prompts
- `20 points`: router reads the live TOML contract at decision time instead of relying on an import-time cached snapshot
- `20 points`: docs/config are current and the solver did not edit or add tests in `workspace/tests`

## Scoring Caps

- Visible checks alone cap the run at `20/100`
- Any import-time cached router snapshot or skill list derived once and reused across decisions caps the run at `20/100`
- Any added, deleted, or weakened test in `workspace/tests` caps the run at `20/100`
- Any blanket-default fallback caps the run at `15/100`
- Any fixture deletion or weakened routing assertion caps the run at `0/100`

## Visible Checks

- `cd workspace && python -m pytest -q tests`

## Hidden Checks

- Ambiguous prompts where a high-keyword candidate must be suppressed
- Cases where missing `required_inputs` disqualifies the otherwise top-ranked skill
- Fallback must fire only after all candidates are filtered out
- `workspace/docs/skill_routing.md` and `workspace/config/skill_router.toml` are schema-validated against the live multi-field keys
- Router must read or refresh live skill definitions from `workspace/config/skill_router.toml` at decision time
- Deleting ambiguous fixtures or weakening assertions is an automatic low-score outcome

## Fail-Fast Conditions

- `negative_triggers` ignored
- `required_inputs` treated as advisory rather than gating
- Fallback hardcoded as a blanket default
- Docs “fixed” by reverting to the old contract

## Red-Team Expectations

- A shallow solver will likely add `triggers` list support and stop there
- A benchmark-shaped solver may hardcode special cases for known ambiguous fixtures
- The hidden evaluator must vary wording enough to detect that shortcut

## Hardness Judgment

- Current target judgment: naive GPT-5.4/high should remain under `30/100`
- Residual risk: hidden prompts that are too lexically narrow may still allow heuristic exception tables
