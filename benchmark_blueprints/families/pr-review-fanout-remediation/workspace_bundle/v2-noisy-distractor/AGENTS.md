# Agent Instructions — `pr-review-fanout-remediation`

## Task

You are on a draft PR branch with exported review threads. Classify every visible
thread before editing code. Only three threads are actionable. One thread is a
duplicate of the reopened preview-fallback regression, and one suggestion points
at an outdated diff hunk that should not be applied on the current branch.

## Inputs

- `src/policy/approval_router.py`
- `src/policy/preview.py`
- `tests/test_preview.py` — writable regression test surface
- `tests/test_approval_router.py` — visible read-only test
- `docs/approval_policy.md`
- `.github/review_export/pr_214_threads.json`
- `.github/review_export/pr_214_summary.md`
- `artifacts/review/thread_digest.md` — stale, non-authoritative
- `.scenario_variant`
- optional `release_context/` and `incident_context/`

## Required Outputs

- update `src/policy/approval_router.py`
- update `src/policy/preview.py`
- update `tests/test_preview.py`
- update `docs/approval_policy.md`
- write `artifacts/review/thread_triage.json`
- write `artifacts/review/reply_draft.md`
- write `artifacts/review/remediation_report.md`

## Rules

- Classify threads before editing code.
- Use exact thread ids from the review export.
- Treat `thread_digest.md` as a hint, not authority.
- Do not modify `.github/review_export/`, `artifacts/review/thread_digest.md`,
  `tests/test_approval_router.py`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`,
  `release_context/`, or `incident_context/`.
- Do not add `legacy_preview_hint`.
- Do not use the network.

## What Good Looks Like

- `thread_triage.json` marks the fallback issue actionable, the duplicate thread
  as duplicate, and the outdated hunk as obsolete.
- `build_policy_preview(..., preview_enabled=False, approval_state="manual_review", ...)`
  keeps `approval_state="human_review_required"` and `requires_human_review=true`.
- the writable preview test covers the fallback case.
- docs show the corrected fallback payload.
- replies map each thread id to the change that was made, duplicated, or
  intentionally not applied.
