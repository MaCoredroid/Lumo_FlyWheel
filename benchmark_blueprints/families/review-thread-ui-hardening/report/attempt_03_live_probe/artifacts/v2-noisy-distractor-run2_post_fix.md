# Post-fix Verification

- Fixed reopen: `RT-118`
- Route: `/pull/241/review-thread`
- Reopen viewport: `ios-review-reopened` (`390x844`)
- Runtime change: on mobile, the metadata row now wraps so the title can break onto its own line and the trailing action cluster no longer forces clipping.
- Accessibility change: the actionable `reply-thread-menu` control now has the accessible name `Open reply thread actions`.
- Verification run: `python3 repo/tests/test_review_thread_ui.py`
