# Evaluator Contract

- `family_id`: `backlog-decomposition-scheduler`
- `task_id`: `backlog-decomposition-scheduler/dependency-schedule`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Scoring Breakdown

- `20` points: Dependency correctness
  - blocked work is not scheduled before prerequisites
  - critical-path items are ordered coherently
- `20` points: Capacity and resource compliance
  - scarce specialists are not double-booked
  - concurrency assumptions fit the runtime capacity fixture
- `15` points: Risk isolation and regression avoidance
  - observability or guardrail work lands before risky rollout work
  - high-risk sequencing does not front-load avoidable regressions
- `20` points: Objective-delta quality
  - the ordering advances the stated near-term objective
  - the plan does more than maximize raw throughput
- `15` points: Evidence and assumption discipline
  - checked facts are separated from inferred assumptions
  - the solver does not claim fixture-backed validation that was not available
- `10` points: Partial-progress milestone score
  - key prerequisite prefix is correct even if the full schedule is imperfect
  - milestone placement is directionally right under the hidden runtime instance

## Caps

- Cap at `20/100` if the attempt is not grounded in concrete runtime fixtures from the provided bundle or if it invents backlog identifiers, capacities, or rollout gates as though they were observed.
- Cap at `20/100` if the attempt claims gold-schedule, test, or fixture validation without the runtime files being present.
- Cap at `25/100` if the schedule is strategically coherent but remains placeholder-based, generic, or assumption-heavy.
- Cap at `15/100` if risky rollout work is placed ahead of observability or prerequisite work.
- Cap at `10/100` if the response is generic advice rather than a concrete ordered schedule.

## Automatic Failure Conditions

- `0-5` if the submission omits an ordered schedule.
- `0-5` if the submission fabricates passed checks or concrete fixture contents.
- `0-10` if the submission ignores dependency ordering entirely.

## Evidence Rules

- Only the provided runtime bundle counts as grounding evidence; no live web, live SaaS, or unrelated workspace evidence should improve the official score.
- Multiple schedules may be acceptable; full scoring should be based on constraint satisfaction and objective progress, not exact string equality alone.
- When runtime fixtures are absent, explicit uncertainty can still earn evidence-discipline points, but it cannot earn full grounding points.
- Partial-progress credit should come from correct milestone placement, not from verbose generic rationale.

## Judging Procedure

1. Verify the submission includes an ordered schedule plus an assumption ledger.
2. Score the six categories above from the solver output against the held-out runtime instance.
3. Apply caps before rounding the final score.
4. Record the raw score, applied cap, and final score in `benchmark_run.md`.

## Meaningfulness Check

- A valid runtime instance must include at least one high-value item blocked by prerequisites, one scarce-specialist bottleneck, and one risky rollout gated on observability or guardrail work.
- If a family-bundle-only attack can exceed `30/100` without the held-out runtime fixtures, the benchmark is under-hardened and should be revised.
