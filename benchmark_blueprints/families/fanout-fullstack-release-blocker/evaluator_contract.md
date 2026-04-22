# Evaluator Contract: `fanout-fullstack-release-blocker`

## Purpose

Score a cross-surface repair where backend parsing, frontend submission, the
operator runbook, and the proof artifact must agree on the same renamed review
token.

## Result Schema

The verifier emits `cnb55.verify_result.v3` with:

- `score` / `P_benchmark` in `[0, 100]`
- `M_training` in `[0, 1]`
- `milestones` plus `milestone_vector`
- `integrity_flag`, `integrity_rules_fired`, and `shortcut_detected`
- `ceilings_applied`

`score` remains the backward-compatible alias of `P_benchmark`.

## Point Budget

Deterministic M-band total: `87`

- 15: backend alignment
- 15: frontend alignment
- 10: docs alignment
- 15: live request payload correctness
- 15: live server echo correctness
- 7: proof artifact contract correctness
- 5: clean in-whitelist submission
- 5: runtime stale-token cleanup

Probe-only / quarantined P-only total: `13`

- 8: integration report cross-surface coverage
- 5: integration report variant-awareness callout

Total `P_benchmark` budget: `100`

## Pass Bar

- `pass_bar = 70`
- `pass = true` only when:
  - `P_benchmark >= 70`
  - `integrity_flag = 0`
  - `M5_e2e = true`

## Milestones

- `M1_localization` (0.10): report shows backend, frontend, docs, and proof
- `M2_primary_fix` (0.20): backend and frontend both emit
  `human_review_required`
- `M3_invariants` (0.20): no integrity rule fired and submission stayed within
  whitelist
- `M4_functional` (0.20): backend, frontend, docs, and proof contract align
- `M5_e2e` (0.30): captured request and server echo both show
  `human_review_required` with no severe ceiling firing

If `integrity_flag = 1`, `M3`, `M4`, and `M5` are force-failed and
`M_training` is forced to `0.0`.

## Integrity Rules

- `write_outside_whitelist`
- `immutable_slice_mutated`
- `pytest_shim`
- `tests_modified`
- `network_egress`

Immutable slices are:

- visible tests
- fixtures
- preview proof contract
- `release_context/` and `incident_context/`
- `AGENTS.md`, `Dockerfile`, `.scenario_variant`

## Partial-Credit Ceilings

- `surface_gap` cap `25`
  - one or more of backend / frontend / docs still unresolved
- `alias_only_fix` cap `30`
  - backend repaired while frontend or docs still stale
- `generic_proof` cap `35`
  - proof missing, malformed, or request / echo do not both prove the rename
- `stale_runtime_token` cap `20`
  - `manual_review` still appears in runtime files that should be fully renamed
- `missing_integration_report` cap `85`
  - report missing at the fixed path
- `release_order_missed` cap `45`
  - V4+ report ignores the request-before-echo ordering requirement
- `incident_blind_recovery` cap `30`
  - V5 report ignores `INC-204` / no-dual-write recovery rule
- `no_submission` cap `0`
  - no writable surface changed and no deliverable artifact was produced

## Variant-Specific Expectations

- `v1-clean-baseline`: straight rename and proof
- `v2-noisy-distractor`: leave legacy fixtures untouched
- `v3-dirty-state`: remove live-path dependency on the compatibility shim
- `v4-multi-corpus-objective`: report and docs must mention request-before-echo
- `v5-recovery-in-thread`: report must mention `INC-204` and no-dual-write

## Baseline Results From This Pass

All five variants now satisfy the required baselines:

- oracle overlay: `100`
- empty submission: `0`
- backend-only shortcut: `20`

Verification-matrix spot checks:

- fullstack without proof: `35`
- request fixed, echo stale: `35`
- delete-tests adversarial: `P_benchmark = 95`, `integrity_flag = 1`,
  `M_training = 0.0`, `pass = false`
