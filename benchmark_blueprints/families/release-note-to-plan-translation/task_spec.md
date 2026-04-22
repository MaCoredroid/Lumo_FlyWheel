# Release Note To Plan Translation

## Task Identity
- `task_id`: `release-note-to-plan-translation/implementation-plan`
- `family_id`: `release-note-to-plan-translation`
- `scenario_type`: `strategic_management`

## Task Prompt
Convert frozen release notes, repo state, and optional release / incident context
into a concrete implementation plan. The plan must identify prerequisite work,
dependency order, a bounded first milestone, and the main user-visible risk of
sequencing the release incorrectly.

## Workspace Bundle
- `release_notes/`
- `repo_inventory/`
- `release_context/` when present
- `incident_context/` when present
- `tests/test_plan_brief.py`
- `bin/cnb55-brief`

## Structured Output Contract
- Write `brief_input.json` at the workspace root.
- Validate with `./bin/cnb55-brief validate brief_input.json`.
- Submit with `./bin/cnb55-brief submit brief_input.json`.
- The canonical schema version is `cnb55.release_plan_brief.v1`.
- The CLI writes `brief/manager_brief.json` and `brief/manager_brief.md`.

## Required Fields
- `first_milestone_id`
- `ordered_steps[]` with `step_id`, `rank`, `title`, `summary`,
  `bounded_deliverable`, and evidence paths
- `dependency_notes[]`
- `primary_risk`
- `assumption_ledger[]` with at least one `missing` row

## Variant Progression
- `v1-clean-baseline`: clean dependency chain, bounded-first-milestone test
- `v2-noisy-distractor`: stale release-note experiment should not anchor the plan
- `v3-dirty-state`: abandoned draft / sunk-cost trap should not become the first milestone
- `v4-multi-corpus-objective`: current objective shifts from speed to enterprise-safe rollout
- `v5-recovery-in-thread`: prior rollback incident invalidates rollout-first ordering

## Saturation And Renewal
If this family's mean `P_benchmark` exceeds 80 for two consecutive training
rounds, the family is due for renewal. Renewal queue:
1. Add a mid-trajectory state-change variant where a staffing or rollout gate changes after the first plan draft.
2. Add a contradictory-evidence variant where two repo-inventory files disagree and the agent must explicitly reconcile them.
