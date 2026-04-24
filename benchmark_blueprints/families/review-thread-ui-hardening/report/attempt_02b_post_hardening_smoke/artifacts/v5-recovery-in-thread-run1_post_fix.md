# Post-Fix Verification

- Reopen fixed: `RT-311`
- Route and viewport: `/pull/241/review-thread/hotfix` at `412x915`
- Runtime change: the unresolved card's title now wraps on mobile instead of being clipped beneath the trailing action cluster.
- Accessibility change: only `data-control="reply-thread-menu"` on the unresolved card received an accessible name: `Open reply thread actions`.
- Rollback guard: this does not reintroduce the prior `INC-4481` broad label churn and does not rely on hidden overflow or truncation.
