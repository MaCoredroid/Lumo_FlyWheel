# Task Spec: `pr-review-fanout-remediation`

## Task Prompt

Address a draft PR that has five visible review threads across code, tests, and docs, but only three are actionable. You may delegate independent remediation work, but you must first classify each thread as `actionable`, `obsolete`, or `duplicate`, and identify any `duplicate_of` relationship. Then implement only the required fixes and prepare concise reviewer-facing replies.

The seeded review export includes one reopened actionable thread, one duplicate thread that refers to the same preview fallback issue, one outdated suggestion that conflicts with the current branch, one missing regression test, and one docs example drift.

## Workspace Bundle

- Feature branch repo with:
  - `src/policy/approval_router.py`
  - `src/policy/preview.py`
  - `tests/test_approval_router.py`
  - `tests/test_preview.py`
  - `docs/approval_policy.md`
  - `.github/review_export/pr_214_threads.json`
  - `.github/review_export/pr_214_summary.md`
  - `artifacts/review/thread_digest.md`, which is intentionally stale and is not authoritative for thread state
- Visible task contract:
  - apply code and docs fixes in the repo
  - prepare `artifacts/review/thread_triage.json` with `{thread_id, disposition, rationale, duplicate_of?}`
  - prepare a machine-readable reply bundle such as `artifacts/review/reply_draft.md`

## Seeded Orchestration Problem

The parent agent should delegate only independent actionable comments. The seeded failure mode is that one reviewer suggestion references an older diff hunk and is now obsolete, one duplicate thread echoes the reopened preview fallback issue, and the stale digest is not a reliable answer key. A naive solver can waste effort "fixing" resolved or outdated comments and still miss the reopened regression.

## Required Surfaces

- `subagents`
- `review`
- `shell`
- `apply_patch`

Preferred:

- `git`
- `worktree`

## Deliverables

- Code and test patch addressing the still-actionable review items
- Docs update for the drifted example
- Reviewer-facing reply draft mapped to thread identifiers
- Short remediation report explaining what was acted on and what was intentionally not changed

## Grader Contract

The task scores well only if the submission passes these gates:

- Phase 1 triage gate: `artifacts/review/thread_triage.json` correctly classifies every visible thread as `actionable`, `obsolete`, or `duplicate`, and any duplicate thread points at the right `duplicate_of` target; no code credit if triage is wrong
- Phase 2 implementation gate: the patch fixes the reopened preview fallback regression, adds or updates the missing regression test requested in review, and updates the docs example that still reflects the old behavior
- Phase 3 reply gate: reviewer-facing replies map changes to the correct actionable or duplicate thread ids and do not claim obsolete suggestions were applied
- Hidden integrity gate: visible tests and review-export artifacts remain intact, and fixes that act on the obsolete hunk or only patch the happy path fail follow-up checks

Low-score behavior includes blanket "all comments addressed" replies, code changes for obsolete comments, or fixes that satisfy tests without resolving the active reopened thread.

## Red-Team Traps

- Treat every visible review comment as actionable
- Ignore thread state and miss the reopened regression
- Write generic reviewer replies with no thread mapping
- Patch tests only to satisfy reviewer wording while leaving runtime behavior wrong
- Update docs copy without changing the code path under review

## Quality Gate

- Probe target: naive GPT-5.4/high should not exceed 30/100 by fixing whichever review comments are easiest to grep.
- Hardening applied after adversarial probe:
  - fixed the duplicate-thread inconsistency so duplicate handling is real, not decorative
  - replaced the shortcut `open_threads.md` idea with a stale digest that is explicitly non-authoritative
  - added a required `thread_triage.json` schema and phase-gated grading before code credit
  - added hidden integrity checks for obsolete-hunk fixes, happy-path-only patches, and review-export tampering
- Probe record:
  - probe model: child GPT-5.4/high adversarial review
  - initial verdict: over 30 too easy because a solver could grep obvious files, trust the digest, and write generic thread-shaped replies
- Current difficulty judgment: under 30 likely for a naive GPT-5.4/high solver after hardening
