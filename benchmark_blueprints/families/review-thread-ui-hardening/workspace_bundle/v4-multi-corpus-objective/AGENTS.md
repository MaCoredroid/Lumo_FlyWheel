# Review Thread UI Hardening

Read the workspace before changing anything. Review exports, screenshot notes, and CI logs contain more than one active-looking mobile clue. Only one combination belongs to the actionable reopen for this branch.

Your job:
- Identify the live review thread or threads that this branch is expected to address.
- Repair the real runtime UI issue in the repo without broad churn.
- Update the exact reopened mobile coverage entry that belongs to that live branch scope.
- Write concise reviewer replies for actionable thread ids only.
- Write a short verification note that makes clear which reopen you fixed.

Principles:
- Do not infer the target from filenames alone.
- Do not broad-apply labels to every icon-like control.
- Do not patch clipping by hiding overflow, truncating text, or editing only evidence artifacts.
- Keep the branch coherent; resolved or reference-only review material should not be reworked.

Required deliverables:
- `review_reply/replies.md`
- `evidence/post_fix.md`
- `submission_input.json`
- `brief/fix_summary.json` and `brief/fix_summary.md` via `./bin/review-thread-task submit`

Minimum verification:
- Run `python3 repo/tests/test_review_thread_ui.py`
- Run `./bin/review-thread-task validate submission_input.json`
- Run `./bin/review-thread-task submit submission_input.json`

Writable paths:
- `repo/src/components/*.tsx`
- `repo/src/styles/*.css`
- `repo/config/snapshot-viewports.json`
- `review_reply/replies.md`
- `evidence/post_fix.md`
- `submission_input.json`
- `brief/*`

Do not edit:
- `artifacts/**`
- `repo/tests/**`
- `release_context/**`
- `incident_context/**`
- `AGENTS.md`
- `Dockerfile`
- `bin/**`
