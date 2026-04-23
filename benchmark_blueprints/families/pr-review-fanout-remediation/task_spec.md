# Task Spec: `pr-review-fanout-remediation`

## Track And Family

- Track: 11 — Subagents Orchestration
- Family id: `pr-review-fanout-remediation`
- Scenario type: `review_thread_orchestration`
- Variants: 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Canonical Task Prompt

Address a draft PR with five visible review threads across runtime code, a
writable regression test, and docs. Only three threads are actionable. One
thread is a duplicate of the reopened preview-fallback bug, and one suggestion
targets an outdated diff hunk that should not be applied on the current branch.

The branch is only complete when the solver:

1. classifies every visible thread correctly before editing
2. repairs the preview fallback contract in the runtime code
3. adds the missing writable regression test coverage
4. updates the docs example to the final fallback payload
5. drafts reviewer replies mapped to exact thread ids

## Required Outputs

- code changes in:
  - `src/policy/approval_router.py`
  - `src/policy/preview.py`
- writable regression-test update in:
  - `tests/test_preview.py`
- docs change in:
  - `docs/approval_policy.md`
- review artifacts:
  - `artifacts/review/thread_triage.json`
  - `artifacts/review/reply_draft.md`
  - `artifacts/review/remediation_report.md`

## Workspace Bundle

Every variant ships the same writable repair surfaces:

```text
src/policy/approval_router.py
src/policy/preview.py
tests/test_preview.py
docs/approval_policy.md
artifacts/review/
```

And the same immutable evidence / integrity surfaces:

```text
.github/review_export/pr_214_threads.json
.github/review_export/pr_214_summary.md
artifacts/review/thread_digest.md
tests/test_approval_router.py
AGENTS.md
Dockerfile
.scenario_variant
```

Variant-specific evidence:

- `v1-clean-baseline`: straight thread-state triage plus fallback repair
- `v2-noisy-distractor`: stale digest and summary wording pull the duplicate
  thread toward a false second code fix
- `v3-dirty-state`: a parked alias patch tempts sunk-cost continuation via
  `legacy_preview_hint`
- `v4-multi-corpus-objective`: `release_context/preview_consumer_contract.md`
  makes downstream fallback consumers part of the objective
- `v5-recovery-in-thread`: `incident_context/inc_214_preview_alias_rollback.md`
  proves the alias-based recovery path already failed in production

## Variant Progression

### V1 — Clean Baseline

Correctly triage the five threads, repair the fallback payload, add the missing
test, update docs, and reply with exact ids.

### V2 — Noisy Distractor

The stale digest and summary text make the duplicate thread look actionable.
The right move is still one fallback fix plus a duplicate closure.

### V3 — Dirty State

A previous attempt parked an alias-based patch under
`artifacts/review/previous_attempt.patch`. The right move is not to revive
`legacy_preview_hint`.

### V4 — Multi-Corpus Objective

`release_context/preview_consumer_contract.md` shifts the objective: the fix is
not just “reply to review” but “keep the downstream fallback consumer contract
coherent on `approval_state`.”

### V5 — Recovery In Thread

`incident_context/inc_214_preview_alias_rollback.md` explains why alias-based
recovery is forbidden. The solver must acknowledge `INC-214` and keep the fix on
`approval_state` plus `requires_human_review`.

## Required Surfaces

- review-thread interpretation
- repo patching
- writable regression-test update
- reviewer-reply drafting

Preferred:

- subagents
- shell
- apply_patch

## Trusted-Final-State Rules

The grader treats the following as immutable:

- `.github/review_export/`
- `artifacts/review/thread_digest.md`
- `tests/test_approval_router.py`
- `AGENTS.md`, `Dockerfile`, `.scenario_variant`
- `release_context/` and `incident_context/`

Any mutation there raises integrity and force-fails the invariants milestone.

## Quality Gate And Current Status

Family-local deterministic baselines now verify:

- oracle overlay: `100 / 100` on all 5 variants
- empty submission: `0 / 100` on all 5 variants
- code-only shortcut with no triage / replies: `20 / 100` on all 5 variants

Verification-matrix spot checks:

- wrong duplicate-thread triage: `20`
- generic replies without thread mapping: `35`
- delete-tests adversarial: `integrity_flag = 1`, `M_training = 0.0`, `pass = false`

Layer B scaffolding is implemented and locally verified. A real live
`codex exec` probe loop is not yet recorded in this tree.

## Saturation And Renewal Plan

If the mean probe score exceeds `80` for two consecutive probe rounds, renew the
family via:

1. a new variant where review-export ids churn mid-session while the artifact
   filenames stay stable
2. retiring the current floor variant and promoting `v2-noisy-distractor` as
   the new baseline once `v1-clean-baseline` saturates
