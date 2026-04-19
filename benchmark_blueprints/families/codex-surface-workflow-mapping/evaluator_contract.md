# Evaluator Contract

- `family_id`: `codex-surface-workflow-mapping`
- `task_id`: `t2_codex_surface_daily_triage_mapping`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Scoring Breakdown

- `30` points: Correctly identifies the live workflow entrypoint and scope.
- `20` points: Skill and TOML artifacts are consistent with each other and with the same entrypoint.
- `20` points: Automation proposal uses the cadence actually supported by repo evidence and keeps task semantics separate from schedule semantics.
- `20` points: `mapping_note.md` cites concrete source paths and exact command literals for each major mapping decision.
- `10` points: Artifacts are specific to the repo rather than generic Codex scaffolding.

## Caps

- Cap at `20/100` if evidence comes from outside the provided benchmark bundle or if command literals are inferred without bundle support.
- Cap at `20/100` if the chosen entrypoint is unsupported or only weakly supported by the bundle.
- Cap at `25/100` if artifacts are internally consistent but could fit many repos.
- Cap at `25/100` if the automation cadence is plausible but not grounded in evidence.
- Cap at `15/100` if the mapping note lacks exact source paths or command literals.
- Cap at `10/100` if artifacts point to inconsistent entrypoints or scopes.

## Automatic Failure Conditions

- `0` if different artifacts point to different workflow entrypoints.
- `0-5` if required artifacts are missing.
- `0-5` if the solver invents commands, schedules, or fixtures not present in the bundle.

## Evidence Rules

- Only the provided benchmark bundle counts as evidence unless the task explicitly names an external surface.
- A reusable stale artifact does not count as proof of the live workflow unless corroborated.
- Command literals are required because generic prose can otherwise look correct.
- Schedule claims need direct support from docs, tests, scripts, or configs.

## Judging Procedure

1. Verify artifact completeness.
2. Score entrypoint correctness and cross-artifact consistency.
3. Apply cadence and mapping-note checks.
4. Apply caps before repo-specificity points.

## Meaningfulness Check

- A valid task instance must contain at least two candidate workflows or entrypoints so that cross-surface resolution matters.
- If one stale artifact can be copied into all outputs and still score above `40`, the authored instance is too easy.
