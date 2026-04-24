# Post-fix Verification

- Reopen fixed: `RT-311`
- Route and viewport: `/pull/241/review-thread/hotfix` at `412x915`
- Runtime change: the unresolved thread title now wraps instead of clipping under the trailing actions, and the action cluster wraps below the title on mobile.
- Accessibility scope: only the unresolved `reply-thread-menu` button received an explicit label (`Reply thread actions`), which avoids the reverted broad-label pattern called out in `INC-4481`.
- Reference-only threads left untouched: `RT-204` and `RT-099`
