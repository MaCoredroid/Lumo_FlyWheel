# Post-fix Verification

- Reopen fixed: `RT-311`
- Route: `/pull/241/review-thread/hotfix`
- Viewport: `412x915` via `release-mobile-recovery`
- Target control: unresolved card `reply-thread-menu`
- Runtime change: the unresolved mobile title now wraps onto its own line and the trailing action cluster moves below it instead of collapsing into the title row.
- Accessibility change: only the unresolved reply-thread menu received `aria-label="Open reply thread actions"`, avoiding the rolled-back broad labeling from `INC-4481`.
- Verification run: `python3 repo/tests/test_review_thread_ui.py`
