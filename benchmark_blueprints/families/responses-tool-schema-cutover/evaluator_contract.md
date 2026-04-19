# Evaluator Contract

## Scope
- Family: `responses-tool-schema-cutover`
- Task: `cnb55-core-responses-tool-schema-cutover-router-replay`
- Runtime class: small Python gateway repo using Responses-style event fixtures

## Evaluation Intent
Measure whether the solver can repair a Responses-native adapter and replay stack without collapsing repeated tool calls, faking chronology, or fixing only docs and config. The evaluator should reward real `call_id`-scoped reasoning and punish visible-fixture-only join logic.

## Required Submission Artifacts
- Code patch touching adapter or reducer logic as needed
- Test changes or additions that protect replay semantics
- `codex/config.toml` alignment to Responses runtime
- Migration note update

## Scoring Model
- `100` total points, awarded by category:
  - `45`: functional repair of adapter and reducer semantics
  - `20`: regression protection and test quality
  - `15`: non-code contract alignment across config and migration note
  - `10`: verification quality and evidence-backed reasoning
  - `10`: robustness against hidden traces and shortcut resistance

## Score Caps
- Cap at `10/100` for docs-only, config-only, or analysis-only submissions.
- Cap at `20/100` if the solver proposes a plausible patch shape but provides no real code-level repair or test evidence.
- Cap at `20/100` if repeated same-name tool calls can still collapse.
- Cap at `25/100` if `call_id` is recognized but chronology remains heuristic or visible-fixture-shaped.
- Cap at `30/100` if only one of adapter or reducer is actually repaired.
- Cap at `50/100` if code is correct but config or docs remain stale.

## Visible Gates
- Run:
  - `pytest -q tests/test_adapter_contract.py tests/test_replay_render.py`
- Required visible scenario:
  - a fixture with two same-name tool calls and out-of-order tool results
  - transcript render must preserve both invocations in order

## Hidden Gates
- Withheld replay where same tool name is invoked multiple times and only `call_id` disambiguates joins
- Withheld chronology case where tool results arrive after other output items
- Differential render oracle for a hidden JSONL trace
- Grep or AST guard ensuring reducer does not dedupe by tool name alone
- Config and docs checks for Responses-native terms such as `call_id` and event items

## Integrity And Shortcut Checks
- Reject if tests were deleted or materially weakened
- Reject if reducer sorts output to hide arrival-order defects
- Reject if `call_id` is reconstructed from visible-fixture ordinals instead of event data
- Reject if docs and config are updated while code remains legacy-shaped

## Variant Hardness Notes
- `V1`: visible repeated-call fixture is enough to expose naive name-based joins
- `V2`: stale replay fixtures and misleading warnings should not affect grading
- `V3`: dirty transcript-fixture edits must be preserved
- `V4`: code, config, and migration note all required
- `V5`: follow-up injects a regression caused by an earlier chronology-blind fix

## Current Hardness Judgment
- Actual recorded solver run: `20/100`
- Naive `gpt-5.4/high` above `30/100`: `unlikely under current rubric`
