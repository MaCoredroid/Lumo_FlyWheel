# Release Note To Plan Translation

## Task Identity
- `task_id`: `release-note-to-plan-translation/implementation-plan`
- `family_id`: `release-note-to-plan-translation`
- `scenario_type`: `strategic_management`

## Task Prompt
Convert frozen release notes, repo state, and test inventory into a concrete implementation plan. The plan must identify prerequisite cleanup, dependency order, user-visible risk, and the smallest meaningful first milestone.

## Workspace Bundle
- `release_notes/`
- `repo_inventory/`
- `tests/plan_contract.py`
- `artifacts/gold_plan_outline.json`

## Seeded Strategic Ambiguity
- One release-note bullet depends on another but is written as if independent.
- The most obvious first milestone is too large.
- A user-visible risk is implied by the repo inventory rather than stated in the notes.

## Required Surfaces
- `shell`
- plan synthesis
- dependency ordering
- concise planning output

## Expected Deliverables
- Ordered implementation plan.
- Milestone breakdown.
- Risk note tied to the release notes and repo inventory.

## Grader Contract
- Visible checks:
  - `pytest -q tests/plan_contract.py`
- Hidden checks:
  - Plan order matches the gold dependency graph.
  - First milestone is appropriately bounded.
  - Risk note captures the withheld user-visible constraint.

## Quality Gate
- Target naive score: `20/100`.
