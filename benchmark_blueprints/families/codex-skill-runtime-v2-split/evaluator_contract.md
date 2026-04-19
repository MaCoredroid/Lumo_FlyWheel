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
- `10`: visible tests pass.
- `10`: smoke command succeeds from the authored bundle.
- `20`: hidden follow-up bundle proves a second skill or heartbeat reuses the canonical structure.
- `20`: hidden stale-reference pack proves monolith and duplicate references are fully retired.
- `20`: hidden duplicate-automation pack proves cleanup is semantic, not cosmetic.
- `10`: trusted canonical-alignment checks pass across skill, config, automation, and runbook.
- `10`: hidden dirty-workspace preservation checks pass.

## Caps
- If the command path is not actually runnable, total score is capped at `20`.
- If duplicate automations remain semantically live, total score is capped at `25`.
- If only the visible bundle is proven and no hidden follow-up reuse pack is exercised, total score is capped at `20`.
- If the fix is mostly docs/config/file creation without canonical alignment, total score is capped at `20`.
- If tests or smoke are bypassed, total score is capped at `0`.

## Passing Interpretation
- Target benchmark-design outcome: a naive GPT-5.4/high solver should land around `20/100`.
- Scores above `30` indicate the family still rewards scaffolding too easily.
