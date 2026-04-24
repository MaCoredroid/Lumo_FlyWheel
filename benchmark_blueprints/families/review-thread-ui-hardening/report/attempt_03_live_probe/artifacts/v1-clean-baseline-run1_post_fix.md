# Post-fix Verification

- Reopen fixed: `RT-118`
- Route: `/pull/241/review-thread`
- Mobile coverage entry updated to `ios-review-reopened` at `390x844`
- Runtime change: the unresolved card wraps the long title onto its own mobile row so the trailing action cluster no longer clips it
- Accessibility change: only the actionable reply control in that reopened cluster is named with `aria-label="Reply to thread"`
- Reference-only material left alone: `RT-099` and the resolved screenshot path were not reworked

Verification:

- `python3 repo/tests/test_review_thread_ui.py`
- `./bin/review-thread-task validate submission_input.json`
- `./bin/review-thread-task submit submission_input.json`
