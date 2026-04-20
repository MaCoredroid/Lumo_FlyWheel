# Backlog Decomposition Scheduler

## Task Identity
- `task_id`: `backlog-decomposition-scheduler/dependency-schedule`
- `family_id`: `backlog-decomposition-scheduler`
- `scenario_type`: `strategic_management`

## Task Prompt
Decompose a frozen backlog into an ordered execution schedule that respects dependency edges, staffing constraints, and risk isolation. Produce a schedule that maximizes near-term objective progress without front-loading avoidable regressions.

## Workspace Bundle
- `backlog/items.json`
- `constraints/team_capacity.json`
- `artifacts/gold_schedule.json`
- `tests/test_schedule_contract.py`

## Seeded Strategic Ambiguity
- A high-value item is blocked by two smaller prerequisites.
- One tempting parallelization path would overload a constrained specialist.
- A risky item should be delayed until after an observability task lands.

## Required Surfaces
- `shell`
- dependency scheduling
- capacity reasoning
- concise planning output

## Expected Deliverables
- Ordered backlog schedule.
- Dependency explanation.
- Risk-isolation note.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_schedule_contract.py`
- Hidden checks:
  - Dependency ordering matches the gold schedule constraints.
  - Capacity limits are respected.
  - Risk-isolation sequencing is correct.

## Quality Gate
- Target naive score: `20/100`.
