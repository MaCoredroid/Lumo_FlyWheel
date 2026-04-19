# Evaluator Contract

## Scope
This evaluator scores one solver attempt against `task_spec.md` for the `incident-retro-runbook-closure` family.

## Required Solver Inputs
- `task_spec.md`
- `codex/config.toml`
- `skills/retro-alignment/SKILL.md`

## 100-Point Breakdown
- `25` Authoritative retro extraction
  - uses `retro/action_items.json` as source of truth
  - separates supporting context from authoritative action items
- `20` Three-surface alignment
  - runbook, helper, and automation prompt agree on command and sequence
- `15` Helper-script repair quality
  - retires the stale subcommand as the primary path
  - keeps the change behavior-preserving and narrow
- `15` Automation prompt fidelity
  - prompt text matches verification step and escalation target
  - cadence or destination are not changed gratuitously
- `15` Runbook and follow-up note quality
  - on-call instructions reflect the corrected sequence
  - operator-facing note captures the actual repair
- `10` Constraint discipline
  - no unrelated docs churn
  - no test or expectation-file edits

## Caps
- Cap at `20/100` if the attempt is not grounded in `retro/action_items.json`.
- Cap at `25/100` if fewer than all three surfaces (`runbook`, `helper`, `automation prompt`) are repaired.
- Cap at `15/100` if the stale and current commands are both left as acceptable operator paths.
- Cap at `10/100` if the attempt updates only docs, notes, or TOML fields without helper behavior repair.

## Scoring Procedure
1. Score each category from the solver's written attempt output.
2. Apply all caps after totaling the raw score.
3. Record raw score, caps, and final score in `benchmark_run.md`.

## Evidence Rules
- Full extraction points require naming `retro/action_items.json` as authoritative.
- Full alignment points require a consistent command and escalation target across all three surfaces.
- Do not award automation points when the solver discusses only schedule fields and ignores prompt text.
- Do not award helper points when the stale subcommand remains the primary operator path.

## Hardness Target
- Calibration target for a naive `GPT-5.4/high` solver: about `20/100`
- Upper guardrail: if a naive solver appears to score above `30/100`, harden the task and rerun
- Lower guardrail: if a serious solver falls below `10/100` because the blueprint is incoherent, clarify the task and rerun
