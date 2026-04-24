# PR Intent Regression Review

- `family_id`: `pr-intent-regression-review`
- `task_id`: `t2_pr_review_markdown_registry_regressions`
- `scenario_type`: `contextual_code_review`

## Task Prompt

Review a feature PR that claims to add Markdown export support for a release-readiness CLI without changing existing JSON behavior. You are given the repo snapshot after the PR, the PR description, the unified diff, a flattened diff view, and a CI snapshot. Do not fix code. Produce only high-signal review findings.

Use the family-local structured-output CLI:

- `bin/cnb55-pr-review schema`
- `bin/cnb55-pr-review validate review_input.json`
- `bin/cnb55-pr-review submit review_input.json`

`submit` writes the canonical deliverables:

- `artifacts/review_packet.json`
- `artifacts/review_findings.md`
- `artifacts/review_summary.md`

## Structured Output Contract

`review_input.json` must use schema version `cnb55.pr_review.v1` and contain:

- `schema_version`
- `variant_id`
- `intent_verdict`
  Allowed values: `matches_intent`, `partial_match`, `does_not_match_intent`, `insufficient_evidence`
- `summary`
- `findings`
  2-4 findings only. Each finding must include:
  - `title`
  - `severity`
  - `changed_file`
  - `changed_lines.start`
  - `changed_lines.end`
  - `linked_surface`
  - `impact`
  - `evidence_paths`
  - optional `test_gap`

The grader reads only `artifacts/review_packet.json`. Markdown files are human-readable mirrors.

## Workspace Layout

Each variant ships:

- `.scenario_variant`
- `AGENTS.md`
- `Dockerfile`
- `bin/cnb55-pr-review`
- `repo/`
  - `src/release_readiness/...`
  - `docs/...`
  - `tests/...`
- `review/`
  - `pr_description.md`
  - `patch.diff`
  - `flattened_diff.md`
  - `ci_snapshot.md`
- `artifacts/`

Variant-specific additions:

- `v2-noisy-distractor`: generated fixture churn and docs churn in the review bundle
- `v3-dirty-state`: `review/stale_draft_comments.md`
- `v4-multi-corpus-objective`: `release_context/nightly_export_consumer.md`
- `v5-recovery-in-thread`: release context plus `incident_context/inc_241_markdown_rollout_rollback.md`

## Variant Progression

- `v1-clean-baseline`
  Clean diff. The two real regressions and the missing-test gap are all present with minimal distraction.
- `v2-noisy-distractor`
  Same core regressions, but diff noise is larger because markdown fixture churn and docs edits are mixed in.
- `v3-dirty-state`
  Same core regressions plus a stale prior review draft that covers only the two obvious code issues. A strong review must not simply complete that partial draft; it still needs the concrete missing-test blocker.
- `v4-multi-corpus-objective`
  Same code changes, but release context shows a downstream nightly consumer that shells out without `--output`.
- `v5-recovery-in-thread`
  Same code changes, plus an incident note showing the default-output regression already caused a rollback once. A review that re-identifies the code bug but misses the recurrence/rollback framing should score as a hard-floor miss.

## Required Surfaces

- `review/patch.diff`
- `review/pr_description.md`
- `review/ci_snapshot.md`
- impacted files under `repo/`
- `release_context/` for V4-V5
- `incident_context/` for V5

No code edits are allowed. The writable surface is the review deliverable only.

## Solver Guidance

- Start from changed lines, then prove impact against an unchanged dependent surface.
- Treat visible CI as a weak signal, not proof of safety.
- Missing tests count only when tied to a concrete behavioral regression path.
- If the bundle does not support a claimed finding, do not fabricate it.

## Saturation And Renewal

This family saturates when mean `P_benchmark > 80` for two consecutive probe rounds on the default probe tier. Renewal queue:

1. add a new variant where a second downstream consumer depends on explicit `--output json`
2. rotate the dirty-state artifact so stale-draft parroting cannot be hardcoded

## Current Status

- Layer A probe acceptance: pending live probe
- Layer B assetization: family-local scorer, manifests, milestones, and verification matrix are expected to exist after regeneration
