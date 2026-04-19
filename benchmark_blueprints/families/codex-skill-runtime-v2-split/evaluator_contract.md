# `codex-skill-runtime-v2-split` Evaluator Contract

## Evaluation Goal
Score whether a solver turns a monolithic Codex runtime bundle into a canonical, executable skill/config/automation bundle instead of creating plausible but disconnected files.

## Visible Checks
- `pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py`
- `python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md`

## Trusted Checks
- No live references remain to the monolithic prompt path.
- Exactly one canonical automation controls the workflow.
- Config and runbook point at real runnable paths.

## Hidden Checks
- Follow-up additions reuse the structured bundle instead of cloning stale text.
- Dirty-workspace sentinel files remain untouched.
- Duplicate automation cleanup is semantic, not cosmetic.

## 100-Point Breakdown
- `15`: visible tests pass.
- `15`: smoke command succeeds from the authored bundle.
- `15`: canonical skill bundle exists and points at a runnable path.
- `15`: config references only canonical live skill paths.
- `15`: only the canonical automation remains active for the task path.
- `10`: runbook, skill, config, and automation are aligned.
- `10`: hidden follow-up reuse checks pass.
- `5`: dirty-workspace preservation checks pass.

## Caps
- If the command path is not actually runnable, total score is capped at `20`.
- If duplicate automations remain semantically live, total score is capped at `25`.
- If the fix is mostly docs/config/file creation without canonical alignment, total score is capped at `30`.
- If tests or smoke are bypassed, total score is capped at `0`.

## Passing Interpretation
- Target benchmark-design outcome: a naive GPT-5.4/high solver should land around `20/100`.
- Scores above `30` indicate the family still rewards scaffolding too easily.
