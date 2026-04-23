# `dead-flag-reachability-audit` Task Spec

- `family_id`: `dead-flag-reachability-audit`
- `task_id`: `t2_dead_flag_preview_reachability_audit`
- Track: `02 Codebase Understanding`
- Structured output schema: `cnb55.flag_audit.v1`

## Task Prompt

Audit the preview rollout flags in this repo and determine which are `live`,
`partial`, or `dead`. The operator request always names these three controls:

- `ENABLE_SHADOW_PREVIEW`
- `ENABLE_PREVIEW_V2`
- `PREVIEW_FORCE_LEGACY`

You must trace defaults, environment parsing, runtime branching, tests, docs,
and any extra release or incident context that exists in the variant before you
classify anything.

Do not patch runtime logic. This family scores reachability judgment, not code
cleanup execution.

## CLI Contract

Every variant ships the same CLI surface:

```bash
./bin/cnb55-flag-audit schema
./bin/cnb55-flag-audit validate brief_input.json
./bin/cnb55-flag-audit submit brief_input.json
```

Expected solver flow:

1. Author `brief_input.json` at the workspace root.
2. Run `./bin/cnb55-flag-audit validate brief_input.json`.
3. Run `./bin/cnb55-flag-audit submit brief_input.json`.

Successful `submit` writes:

- `brief/flag_audit.json`
- `artifacts/flag_audit.md`
- `artifacts/reachability_matrix.json`
- `artifacts/cleanup.patchplan.md`

## Required Output Shape

`brief_input.json` must include:

- `schema_version`
- `variant_id`
- `flags`: exactly three rows, one per named control
- `summary`
- `cleanup_plan`
- `assumption_ledger`

Each flag row must include:

- `flag`
- `status`: one of `live`, `partial`, `dead`
- `alias_of`
- `parser_symbol`
- `runtime_branch_symbol`
- `evidence`
- `disproved_false_positive_path`
- `rationale`

## Variant Progression

### V1 clean baseline

- Core reachability split only.
- `ENABLE_SHADOW_PREVIEW` is the live runtime control.
- `ENABLE_PREVIEW_V2` is accepted by the parser as a legacy alias.
- `PREVIEW_FORCE_LEGACY` is parser-visible but not runtime-live.

### V2 noisy distractor

- Adds a stale 2025 rollout memo that still names `ENABLE_PREVIEW_V2`.
- Tests and docs are now noisy enough that grep-only reasoning should fail.

### V3 dirty state

- Adds a prior-session audit draft plus an unfinished patch that would revive a
  `PREVIEW_FORCE_LEGACY` runtime branch.
- The right answer rejects abandoned intent as proof of current reachability.

### V4 multi-corpus objective

- Adds `release_context/` with release-freeze guidance.
- Classification does not change, but cleanup sequencing must shift toward
  telemetry or docs-first handling for the alias during the freeze.

### V5 recovery in thread

- Adds `incident_context/` showing a rollback after premature alias removal.
- The audit must acknowledge the rollback before recommending any follow-up
  cleanup on `ENABLE_PREVIEW_V2`.

## Workspace Layout

Every variant contains the same top-level surfaces:

- `config/defaults.toml`
- `src/preview/config.py`
- `src/preview/runtime.py`
- `src/preview/service.py`
- `src/preview/legacy.py`
- `src/preview_cli.py`
- `docs/`
- `tests/`
- optional `release_context/`
- optional `incident_context/`
- optional `repo_evidence/`

The solver may write only:

- `brief/*`
- `artifacts/*`
- `brief_input.json`

Any other file mutation is an integrity failure.

## What Full Credit Looks Like

The solver:

- classifies all three flags correctly
- distinguishes parser presence from runtime reachability
- treats `ENABLE_PREVIEW_V2` as alias-only, not as a standalone live branch
- proves `PREVIEW_FORCE_LEGACY` lacks a live runtime branch
- cites at least four distinct surface roots
- names one disproved false-positive path per flag
- keeps the cleanup plan narrow and variant-appropriate

## Partial-Credit Ceilings

The scorer applies hard caps after aggregation:

- `no_audit_file` -> `0`
- `malformed_audit` -> `10`
- `runtime_branch_missing` -> `20`
- `parser_equals_runtime` -> `20`
- `cleanup_overreach` -> `20`
- `alias_collapse` -> `25`
- `incident_blind_cleanup` -> `25`
- `grep_only_reasoning` -> `30`
- `stale_docs_truth` -> `30`
- `sunk_cost_finish` -> `30`
- `objective_drift` -> `35`
- `classification_miss` -> `45`

## Saturation And Renewal

This family should be renewed if `mean P_benchmark > 80` for two probe rounds.
Current renewal queue:

- add deploy-manifest evidence so alias migration can be proven directly
- add a parser-refactor diff that tempts solver over-reliance on intended
  future state
- retire or replace the current V1 if it stops discriminating at all
