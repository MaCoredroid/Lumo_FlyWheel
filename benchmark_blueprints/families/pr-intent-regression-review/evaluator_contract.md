# Evaluator Contract

- `family_id`: `pr-intent-regression-review`
- `task_id`: `t2_pr_review_markdown_registry_regressions`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Point Breakdown

Deterministic M-band (`90` points):

- `20`: identify the default-output regression in `repo/src/release_readiness/cli.py`
- `20`: identify the explicit-JSON registry regression in `repo/src/release_readiness/renderers/registry.py`
- `10`: identify the missing regression-test gap in `repo/tests/test_markdown_export.py`
- `8`: correct changed-line anchors on at least two findings
- `8`: at least two findings link changed code to an unchanged invariant-bearing surface
- `5`: evidence discipline stays inside the provided repo/review/context bundle
- `5`: contextual grounding cites bundle evidence beyond the changed file itself
- `6`: schema/version validity for the structured review packet
- `4`: canonical artifact presence
- `4`: finding-count contract satisfied

Probe-only P-band (`10` points):

- `5`: summary verdict correctly says the PR does not match its stated intent
- `5`: summary explicitly names the JSON/default or explicit-json regressions plus the missing-test gap

## Hard Caps

- cap at `10` for style-only review
- cap at `15` for generated-fixture fixation
- cap at `20` for out-of-bundle evidence or fully wrong/unanchored findings
- cap at `25` when the review never links changed code to unchanged dependent behavior
- cap at `30` when the review leans on passing CI to justify merge safety
- cap at `20` on V3 when the review parrots `review/stale_draft_comments.md`
- cap at `35` on V4-V5 when the review ignores `release_context/nightly_export_consumer.md`
- cap at `30` on V5 when the review fails to acknowledge recurrence or rollback risk from `incident_context/inc_241_markdown_rollout_rollback.md`

## Baselines

- Oracle: `>= 90`
- Empty: `0`
- Shortcut style-only review: `<= 30`

## Integrity Rules

- writing outside `artifacts/*` and `review_input.json`
- mutating immutable slices: `repo/`, `review/`, `release_context/`, `incident_context/`, `AGENTS.md`, `Dockerfile`, `bin/`
- adding pytest shim files
- mutating `repo/tests`
- network egress marker present

Integrity sets `H=1`, force-fails M3/M4/M5, and zeroes `M_training`.

## Meaningfulness Check

A valid instance requires all of the following:

- the diff can support at least two materially different blocking findings
- at least one finding must connect a changed line to an unchanged cross-surface contract
- visible CI alone must be insufficient to clear `30/100`
- the solver must be able to refuse unsupported claims without being forced into fabrication
