# Benchmark Run

- `family_id`: `codex-surface-workflow-mapping`
- `task_id`: `t2_codex_surface_daily_triage_mapping`
- `run_date`: `2026-04-18`
- `agent_id`: `019da332-4976-7222-ab87-540264c10017`
- `model`: `gpt-5.4`
- `reasoning_effort`: `high`
- `run_context`: family directory only, with no task-local workflow bundle present

## Actual Attempt

The child agent produced all four required artifacts and kept them internally consistent. It did not leave the family directory, but it inferred the canonical entrypoint, command wrapper, and automation cadence from the benchmark metadata because the actual script, Make target, docs, fixtures, and stale draft were missing.

This is exactly the sort of polished-but-under-grounded output the family is meant to detect.

## Scoring Against Final Evaluator

- `8/30`: chose a plausible entrypoint, but only from metadata
- `18/20`: artifacts were highly consistent with one another
- `2/20`: automation cadence was explicitly low-confidence and unsupported by bundle evidence
- `8/20`: mapping note cited source paths and command literals, but those literals were inferred rather than bundle-validated
- `5/10`: repo specificity was decent, though still built on placeholders

Raw subtotal: `41/100`

## Caps Applied

- Unsupported-entrypoint and unsupported-command cap from [evaluator_contract.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/evaluator_contract.md): `20/100` maximum when command literals and cadence are not grounded in the provided bundle

Final score: `20/100`

## Hardening From This Run

- Added an explicit provided-bundle-only evidence rule to [task_spec.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/task_spec.md), [evaluator_contract.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/evaluator_contract.md), and [codex/config.toml](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/codex/config.toml).
- Kept the exact-command and cross-artifact-consistency requirements because they force useful structure even in a losing run.

## Final Judgment

- Meaningful after hardening: `yes`
- Naive GPT-5.4/high expected score from a family-bundle-only attack: `about 20/100`
- Target met: `yes`

