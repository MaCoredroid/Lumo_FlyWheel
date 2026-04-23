# `dead-flag-reachability-audit` Evaluator Contract

- `family_id`: `dead-flag-reachability-audit`
- `task_id`: `t2_dead_flag_preview_reachability_audit`
- `verify_result.schema_version`: `cnb55.verify_result.v3`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Evaluation Goal

Score whether the solver can separate parser visibility, runtime reachability,
alias compatibility, and cleanup scope for three confusable rollout flags.

## Visible Checks

Visible budget: `34` raw points.

| Check | Points |
| --- | ---: |
| `brief/flag_audit.json` exists | 4 |
| `brief/flag_audit.json` parses | 6 |
| rendered outputs exist under `artifacts/` | 4 |
| all three flag rows are present | 6 |
| evidence paths resolve inside the bundle | 4 |
| `summary` present | 3 |
| `assumption_ledger` includes one `missing` row | 3 |
| `cleanup_plan` present with at least two actions | 4 |

## Hidden Checks

Hidden deterministic budget: `89` raw points before caps.

| Check | Points |
| --- | ---: |
| correct `status` for each flag | 30 |
| correct parser/runtime symbols per flag | 12 |
| correct alias handling for `ENABLE_PREVIEW_V2` | 12 |
| correct dead-vs-runtime absence for `PREVIEW_FORCE_LEGACY` | 8 |
| evidence spans at least four surface roots | 8 |
| correct disproved false-positive path per flag | 9 |
| cleanup actions align with the variant contract | 10 |

## P-only Quarantine

`cleanup_precision_bonus` contributes `6` points to `P_benchmark` only and is
excluded from `M_training`. This keeps the training band purely deterministic.

## Hard Ceilings

These caps are applied after raw aggregation:

| Ceiling | Max points | Trigger |
| --- | ---: | --- |
| `no_audit_file` | 0 | `brief/flag_audit.json` missing |
| `malformed_audit` | 10 | wrong schema or invalid deliverable |
| `runtime_branch_missing` | 20 | runtime proof absent for the live or partial controls |
| `parser_equals_runtime` | 20 | parser presence treated as enough proof of runtime behavior |
| `cleanup_overreach` | 20 | cleanup actions outrun the evidence |
| `alias_collapse` | 25 | `ENABLE_PREVIEW_V2` treated as a standalone live control |
| `incident_blind_cleanup` | 25 | V5 cleanup ignores the rollback context |
| `grep_only_reasoning` | 30 | evidence lacks enough surface breadth or false-positive handling |
| `stale_docs_truth` | 30 | V2 stale memo cited as current truth without being called stale |
| `sunk_cost_finish` | 30 | V3/V5 unfinished patch treated as current runtime |
| `objective_drift` | 35 | V4 cleanup ignores the release-freeze objective |
| `classification_miss` | 45 | one or more flag statuses are wrong |

## Integrity Rules

Any of these set `integrity_flag = 1`, zero `M3`/`M4`/`M5`, and force score `0`:

- writes outside `brief/*`, `artifacts/*`, or `brief_input.json`
- readonly tree mutation
- pytest shim creation
- visible-test mutation
- `.network_egress_detected` marker present

## Milestones

- `M1_localization`: evidence spans enough roots to prove code-path inspection
- `M2_primary_fix`: `brief/flag_audit.json` exists, parses, and rendered outputs exist
- `M3_invariants`: no integrity failure
- `M4_functional`: all three flag rows plus parser/runtime symbols plus cleanup plan
- `M5_e2e`: all classifications correct and no ceiling `<= 35` fired

## Oracle / Empty / Shortcut Baselines

Observed after regen on `2026-04-22`:

| Input | Expected result |
| --- | --- |
| Oracle | `100` |
| Empty | `0` |
| Shortcut alias/runtime collapse brief | `20` |

The shortcut brief intentionally treats `ENABLE_PREVIEW_V2` as standalone-live
and promotes the abandoned `PREVIEW_FORCE_LEGACY` patch as partial runtime
evidence.

## Meaningfulness Check

The family is only valid if all of these are simultaneously true:

- parser and runtime evidence disagree for at least one flag
- at least one false-positive path is plausible from docs or tests alone
- cleanup scope changes across variants without changing the underlying live or
  partial or dead ground truth
- the scorer can separate alias reasoning from standalone liveness
