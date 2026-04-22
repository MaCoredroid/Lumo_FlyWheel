# Evaluator Contract

- `family_id`: `backlog-decomposition-scheduler`
- `task_id`: `backlog-decomposition-scheduler/dependency-schedule`
- `total_points`: `100`
- `schema_version`: `cnb55.verify_result.v3`

## Score Breakdown

- `18` points: dependency ordering
- `12` points: scarce-role capacity protection
- `10` points: rollout risk gating
- `10` points: objective alignment
- `8` points: slot accuracy versus the gold schedule
- `6` points: evidence grounding
- `8` points: critical-path prioritization
- `38` points: structural / deliverable correctness
- `10` points: partial-progress heuristic (`P_only`, quarantined from training)

## Visible Checks Budget

Visible checks are capped below `30` points and only validate:

- brief exists
- JSON parses
- every backlog item appears exactly once
- slots are contiguous
- risk gate is present
- assumption ledger contains at least one `missing` row

## Hidden Checks Budget

Hidden checks cover at least `52` points via:

- dependency-edge satisfaction
- scarce-role collision detection
- risky rollout ordering
- current-objective alignment
- exact / near-exact slot placement
- variant-gated traps

## Partial-Credit Ceilings

- `no_schedule_file` → `0`
- `malformed_brief` → `10`
- `schedule_without_grounding` → `25`
- `ignored_dependency_blocker` → `30`
- `overloaded_scarce_specialist` → `30`
- `risky_rollout_frontloaded` → `25`
- `stale_priority_anchor` → `35`
- `sunk_cost_finish` → `40`
- `objective_drift` → `45`
- `incident_blind_reselect` → `30`

## Shortcut / Empty / Oracle Baselines

- Empty brief must score `0`.
- Oracle brief must score at least `90`.
- The family shortcut trajectory must score at most `30`.

## Integrity Rules

Rule ids must match scorer call sites 1:1:

- `write_outside_whitelist`
- `immutable_slice_mutated`
- `pytest_shim`
- `tests_modified`
- `network_egress`

## Dual-Band Emission

- `score` remains the backward-compatible alias of `P_benchmark`
- `P_benchmark` is the full 0–100 benchmark score
- `M_training` is the deterministic-only normalized score in `[0, 1]`
- quarantined `P_only` points are limited to the partial-progress heuristic

## Milestones

- `M1_localization` weight `0.10`
- `M2_primary_fix` weight `0.20`
- `M3_invariants` weight `0.20`
- `M4_functional` weight `0.20`
- `M5_e2e` weight `0.30`

## Verification Matrix Requirement

Ship:

- `verification_matrix.md` for V1
- `verification_matrix_v4.md` for one stress variant

Both must cover 6 trajectories:

- Oracle
- Empty
- RAWR grounding stripped
- Shortcut trajectory
- Near-miss trajectory
- Delete-tests adversarial
