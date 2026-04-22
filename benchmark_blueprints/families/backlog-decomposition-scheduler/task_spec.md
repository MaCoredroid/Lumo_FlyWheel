# `backlog-decomposition-scheduler` Task Spec

**Track:** 10 — Strategic Management & Long-Horizon Evolution
**Family id:** `backlog-decomposition-scheduler`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Task Prompt

You are acting as the delivery manager for a backlog that cannot simply be
sorted by raw business value. Some work is blocked by prerequisites, two items
compete for the same scarce migration specialist, and the customer-visible
cutover must not ship before the rollout guardrails exist.

Read the backlog items and supporting evidence, then produce a structured
schedule using the family-local CLI:

```bash
./bin/cnb55-schedule schema
./bin/cnb55-schedule validate brief_input.json
./bin/cnb55-schedule submit brief_input.json
```

The CLI writes the canonical deliverable at `brief/schedule_brief.json` and a
human-readable rendering at `brief/schedule_brief.md`.

The brief must:

1. Name the current objective in `objective_focus`.
2. Schedule every backlog item exactly once with contiguous slots starting at
   `1`. Multiple items may share a slot if they can safely run in parallel.
3. Keep dependency-blocked work after its prerequisites.
4. Keep scarce-role items out of the same slot.
5. Place risky rollout work after its required observability / dry-run gates.
6. Include a scarce-role note, a rollout-gate note, and an assumption ledger
   with at least one `status: "missing"` row.

## Scenario Type

`strategic_management`

## Required Surfaces

- `shell`
- file inspection of `backlog/`, `repo_evidence/`, `release_context/`,
  `incident_context/`
- structured-output CLI usage

No network, browser, MCP, or sub-agents are required to solve the task.

## Workspace Bundle

Every variant ships:

```text
AGENTS.md
Dockerfile
.scenario_variant
artifacts/README.md
bin/cnb55-schedule
backlog/B*.md
repo_evidence/*.md
tests/test_schedule_brief.py
brief/
```

`release_context/` is present in V4+.
`incident_context/` is present in V5.

## Difficulty Ladder

### v1 — clean baseline

The schedule must unblock the high-value dry-run and then the wave-1 cutover
without double-booking the migration SRE.

### v2 — noisy distractor

Adds a stale Q1 fast-lane memo and a tempting backlog item that looks urgent if
the stale memo is trusted over the current staffing / rollout evidence.

### v3 — dirty state

Adds an abandoned partial patch for the distractor item. The trap is to treat
the partial patch as progress instead of sunk cost.

### v4 — multi-corpus objective

Adds release-context evidence showing the current cycle objective shifted from
throughput-first to reliability-first, so observability moves ahead of the
migration-specialist data prep.

### v5 — recovery in thread

Adds incident evidence showing the distractor path was already rolled back in a
prior attempt. The schedule must acknowledge recovery work before re-entering
that lane.

## Expected Deliverables

- `brief/schedule_brief.json`
  - `schema_version`
  - `variant_id`
  - `objective_focus`
  - `schedule[]` with `item_id`, `slot`, `lane`, `summary`, `citations[]`,
    `constraint_tags[]`
  - `scarce_role_plan`
  - `risk_gate`
  - `assumption_ledger[]`
- `brief/schedule_brief.md`

## Layer A Acceptance Target

The intended freeze-gate probe target is:

- family mean in `[15, 25]`
- no variant mean above `40`
- at least one variant at or below `10`
- monotonic difficulty within the usual ±3 tolerance
- oracle `>= 90`
- empty `= 0`
- shortcut `<= 30`

## Layer B Readiness Notes

This family emits both `P_benchmark` and `M_training`, publishes 5-slot
milestones, ships family-local milestone scripts, declares capability tags and
state-delta rules in `family.yaml`, and includes verification-matrix artifacts
for V1 and one stress variant.

## Saturation And Renewal Plan

Trigger saturation when mean `P_benchmark` exceeds `80` for two consecutive
probe rounds.

Renewal queue:

1. Introduce a mid-run staffing update variant.
2. Introduce a contradictory-evidence variant where release and incident
   guidance disagree and must be flagged explicitly.
3. Retire V1 once it becomes a solved floor-check and promote a harder baseline.
