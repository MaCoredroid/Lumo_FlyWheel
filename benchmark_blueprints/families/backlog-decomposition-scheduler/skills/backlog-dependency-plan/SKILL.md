# Backlog Dependency Plan

Use this skill when a task requires ordering a backlog under dependency, capacity, and rollout-risk constraints.

## Workflow
1. Enumerate the concrete backlog items, dependency edges, scarce roles, and rollout gates that are actually present in the runtime bundle.
2. Identify true blockers and prerequisites before proposing any schedule.
3. Respect scarce specialist capacity and avoid double-booking across concurrent work.
4. Sequence observability and risk-reduction work before high-risk rollout items.
5. Optimize for objective progress, not only raw item count.
6. If runtime fixtures are missing, keep the schedule explicit but mark every inferred identifier, capacity, or risk gate as an assumption.

## Anti-Patterns
- Inventing backlog IDs, capacities, or hidden checks as though they were observed.
- Scheduling blocked work first.
- Ignoring specialist capacity.
- Front-loading risky rollout work without guardrails.
- Claiming gold-schedule or test validation without fixture files.

## Done Signal
- The schedule is dependency-aware, capacity-aware, risk-aware, and honest about what was actually grounded.
