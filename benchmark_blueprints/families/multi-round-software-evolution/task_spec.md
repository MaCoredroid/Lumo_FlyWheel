# Multi-Round Software Evolution

## Task Identity
- `task_id`: `multi-round-software-evolution/evolution-round-plan`
- `family_id`: `multi-round-software-evolution`
- `scenario_type`: `strategic_management`

## Task Prompt
Plan the next round of software evolution from a frozen sequence of prior rounds, partial fixes, regressions, and open objective metrics. Recommend the next round focus, identify what should explicitly not be touched this round, and define a partial-progress score that would count as meaningful movement.

## Workspace Bundle
- `round_history/`
- `objective_metrics/`
- `repo_state/`
- `tests/test_evolution_plan.py`

## Seeded Strategic Ambiguity
- One regression from the prior round looks severe but is already contained.
- A second issue is less visible but blocks future progress more strongly.
- The best next round requires leaving some tempting cleanup work untouched.

## Required Surfaces
- `shell`
- long-horizon reasoning
- round planning
- partial-progress definition

## Expected Deliverables
- Next-round focus memo.
- Explicit do-not-touch list.
- Partial-progress metric definition.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_evolution_plan.py`
- Hidden checks:
  - Round focus matches the gold long-horizon priority.
  - Do-not-touch scope avoids destabilizing nearby work.
  - Partial-progress metric is meaningful and aligned to future evaluation.

## Quality Gate
- Target naive score: `20/100`.
