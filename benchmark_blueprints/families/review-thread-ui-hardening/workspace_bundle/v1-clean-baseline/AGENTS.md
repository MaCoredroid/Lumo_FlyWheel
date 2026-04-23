# Review Thread UI Hardening

Read the workspace before changing anything. The actionable review state is in the exported review bundle, not in filenames.

Goal:
- Fix the live unresolved mobile review feedback in the repo.
- Update the exact reopened mobile viewport coverage.
- Write concise replies for the actionable thread ids only.
- Write a short evidence note tied to the fixed route and viewport.

Required workflow:
1. Inspect `artifacts/review/threads.json`, `artifacts/review/export.md`, and the screenshot notes under `artifacts/screenshots/`.
2. Inspect the repo sources, styles, config, and visible test.
3. Make the narrow fix for the real reviewed surface.
4. Write `review_reply/replies.md`.
5. Write `evidence/post_fix.md`.
6. Run `python3 repo/tests/test_review_thread_ui.py`.
7. Author `submission_input.json`.
8. Run `./bin/review-thread-task validate submission_input.json`.
9. Run `./bin/review-thread-task submit submission_input.json`.

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

Do not broad-apply accessibility labels to unrelated controls. Do not reply to resolved threads. Do not fix mobile clipping by hiding or truncating content.
