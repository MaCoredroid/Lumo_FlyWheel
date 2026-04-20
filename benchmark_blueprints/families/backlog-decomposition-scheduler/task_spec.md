# Backlog Decomposition Scheduler

- `task_id`: `backlog-decomposition-scheduler/dependency-schedule`
- `family_id`: `backlog-decomposition-scheduler`
- `scenario_type`: `strategic_management`

## Task Prompt

Turn a frozen delivery backlog into an execution schedule that respects dependency edges, scarce specialist capacity, and rollout risk. The target outcome is near-term objective progress without scheduling blocked work first, overloading the constrained specialist, or shipping a risky rollout before its observability and prerequisite work land.

Produce:

- an ordered backlog schedule
- a dependency rationale for the ordering
- a specialist-capacity note
- a risk-isolation note
- an assumption ledger that separates observed facts from missing inputs

## Workspace Bundle

- Family authoring bundle only:
  - `task_spec.md`
  - `evaluator_contract.md`
  - `codex/config.toml`
  - `skills/backlog-dependency-plan/SKILL.md`
  - `benchmark_run.md`
- A full runtime task instance is expected to provide:
  - a frozen backlog table with stable item identifiers
  - dependency edges or prerequisite metadata
  - team-capacity and scarce-specialist constraints
  - rollout-risk and observability notes
- Those runtime fixtures are intentionally not committed in this family bundle. The canonical benchmark should stay offline and replayable, with no live web or live service dependency in the official score path.

## Seeded Strategic Ambiguity

- A high-value initiative is blocked by smaller prerequisite work.
- One plausible parallel path is invalid because it double-books a scarce specialist.
- A risky rollout should wait until observability or guardrail work lands.
- Generic prioritization advice is intentionally insufficient; the solver must commit to an ordered schedule while staying honest about missing data.

## Required Surfaces

- `shell`
- `skills`
- structured planning output
- explicit evidence and assumption discipline

## Expected Deliverables

- An ordered schedule with concrete backlog identifiers if the runtime instance provides them.
- A dependency explanation that names the critical blockers.
- A capacity note that explains how specialist overload was avoided.
- A risk-isolation note that explains why the risky rollout is or is not delayed.
- An assumption ledger that marks any inferred backlog items, role capacities, or rollout gates as unverified.

## Grader Contract

- Full credit requires:
  - the schedule is grounded in concrete backlog items from the provided runtime instance
  - prerequisite work appears before blocked work
  - scarce specialist capacity is respected across concurrent work
  - observability or guardrail work lands before the risky rollout
  - the ordering improves objective progress rather than maximizing raw item count
  - the answer clearly separates checked facts from assumptions
- Partial credit if the schedule logic is strategically sound but some dependency, capacity, or rollout-gate detail is unresolved.
- Low credit if the answer is mostly generic prioritization advice or if it invents concrete fixture data as though it were observed.

## Red-Team Traps

- Schedule a blocked high-value item before its prerequisites.
- Overload the scarce specialist because the timeline looks faster on paper.
- Front-load the risky rollout before observability or guardrail work.
- Claim fixture-backed validation or gold-schedule agreement when no runtime fixtures were provided.

## Quality Gate

- The runtime prompt should reveal the business objective but not leak the exact hidden dependency graph or gold schedule.
- Valid runtime instances should include at least one tempting but invalid parallelization path and one risky rollout gated on observability or prerequisite work.
- Hidden grading should use held-out, fixed runtime fixtures so the benchmark is replayable and less vulnerable to contamination or prompt leakage.
- The grader should reward partial progress on milestone placement and constraint satisfaction, not only exact schedule-string matches.
- The family bundle must remain bundle-only: no committed backlog fixtures, gold schedule, variant directories, or local service scaffolding are required here.
- A family-bundle-only attack should top out in the low-20 range because concrete backlog identifiers, capacities, and hidden schedule checks are unavailable.
- Latest family-bundle calibration: `benchmark_run.md` records a real `GPT-5.4/high` `codex exec` attempt on `2026-04-19` that finished at `20/100` after the runtime-fixture grounding cap.
