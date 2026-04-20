# Objective-Driven Repo Improvement

## Task Identity
- `task_id`: `objective-driven-repo-improvement/objective-delta-plan`
- `family_id`: `objective-driven-repo-improvement`
- `scenario_type`: `strategic_management`

## Task Prompt
Given a frozen repository snapshot, objective metric history, and a small set of allowed interventions, choose the next improvement step that yields the best objective delta without destabilizing the repo. Provide a decision note, expected delta, and the main regression risk.

## Workspace Bundle
- `repo_snapshot/`
- `objective_history/`
- `allowed_interventions/`
- `tests/test_objective_plan.py`

## Seeded Strategic Ambiguity
- The highest upside intervention also has the highest regression risk.
- A modest intervention compounds with an already-landed change.
- One appealing intervention is disallowed in the current governance window.

## Required Surfaces
- `shell`
- objective reasoning
- intervention ranking
- concise decision note

## Expected Deliverables
- Selected next intervention.
- Expected objective delta.
- Main regression-risk note.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_objective_plan.py`
- Hidden checks:
  - Selected intervention matches the gold strategic choice.
  - Governance window is respected.
  - Delta estimate and risk note are directionally correct.

## Quality Gate
- Target naive score: `20/100`.
