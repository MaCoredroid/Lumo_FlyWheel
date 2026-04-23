# Evaluator Contract: `pr-review-fanout-remediation`

## Purpose

Score whether a solver triages mixed-state PR review threads correctly, repairs
the reopened preview fallback contract, updates the writable regression test and
docs, and maps reviewer replies back to the exact review-export ids.

## Result Schema

The verifier emits `cnb55.verify_result.v3` with:

- `score` / `P_benchmark` in `[0, 100]`
- `M_training` in `[0, 1]`
- `milestones` plus `milestone_vector`
- `integrity_flag`, `integrity_rules_fired`, and `shortcut_detected`
- `ceilings_applied`

`score` remains the backward-compatible alias of `P_benchmark`.

## Point Budget

Deterministic M-band total: `95`

- `25`: correct thread-state classification
- `10`: correct duplicate mapping for `T-214-02 -> T-214-01`
- `20`: fallback payload contract fixed
- `10`: router normalization preserved on the live preview path
- `10`: writable preview regression test updated
- `10`: docs fallback example aligned
- `5`: reply draft maps exact thread ids
- `5`: remediation report scopes acted-on vs intentionally skipped work

Probe-only / quarantined P-only total: `10`

- `5`: reply specificity names real files and fix surfaces
- `5`: variant-awareness callout in the remediation report

Total `P_benchmark` budget: `100`

## Pass Bar

- `pass_bar = 70`
- `pass = true` only when:
  - `P_benchmark >= 70`
  - `integrity_flag = 0`
  - `M5_e2e = true`

## Milestones

- `M1_localization` (0.10): triage file exists and replies use thread ids
- `M2_primary_fix` (0.20): fallback payload and router normalization satisfy the
  hidden contract
- `M3_invariants` (0.20): no integrity rule fired
- `M4_functional` (0.20): code, writable test, docs, and reply mapping align
- `M5_e2e` (0.30): correct triage and final artifacts close the actionable work
  without reviving obsolete or rolled-back paths

If `integrity_flag = 1`, `M3`, `M4`, and `M5` are force-failed and
`M_training` is forced to `0.0`.

## Integrity Rules

- `write_outside_whitelist`
- `immutable_slice_mutated`
- `pytest_shim`
- `tests_modified`
- `network_egress`

Immutable slices are:

- `.github/review_export/`
- `artifacts/review/thread_digest.md`
- `tests/test_approval_router.py`
- `release_context/` and `incident_context/`
- `AGENTS.md`, `Dockerfile`, `.scenario_variant`

## Partial-Credit Ceilings

- `wrong_triage` cap `20`
  - any thread-state misclassification
- `missing_duplicate_mapping` cap `25`
  - duplicate thread not mapped back to `T-214-01`
- `missing_fallback_fix` cap `30`
  - fallback contract still drops the human-review fields
- `generic_replies` cap `35`
  - reply bundle does not map exact thread ids or dispositions
- `legacy_alias_reintroduced` cap `20`
  - runtime fix revives `legacy_preview_hint`
- `objective_contract_missed` cap `45`
  - V4 report ignores the downstream consumer contract
- `incident_blind_recovery` cap `30`
  - V5 report ignores `INC-214` / no-alias recovery
- `missing_report_scope` cap `80`
  - remediation report missing or does not distinguish acted-on from skipped work
- `no_submission` cap `0`
  - no writable repair surface or required artifact was produced

## Variant-Specific Expectations

- `v1-clean-baseline`: correct triage plus fallback repair
- `v2-noisy-distractor`: duplicate thread must stay duplicate despite stale summary noise
- `v3-dirty-state`: do not finish the parked alias patch
- `v4-multi-corpus-objective`: report the downstream consumer contract on `approval_state`
- `v5-recovery-in-thread`: report `INC-214` and do not revive `legacy_preview_hint`

## Baseline Results From This Pass

All five variants now satisfy the required baselines:

- oracle overlay: `100`
- empty submission: `0`
- code-only shortcut: `20`

Verification-matrix spot checks:

- wrong duplicate-thread triage: `20`
- generic replies without thread mapping: `35`
- delete-tests adversarial: `P_benchmark = 0`, `integrity_flag = 1`,
  `M_training = 0.0`, `pass = false`
