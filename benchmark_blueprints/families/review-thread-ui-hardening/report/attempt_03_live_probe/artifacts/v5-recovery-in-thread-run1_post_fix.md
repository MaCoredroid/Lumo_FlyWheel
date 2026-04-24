# Post-fix verification

- Reopen fixed: `RT-311`
- Route: `/pull/241/review-thread/hotfix`
- Viewport: `412x915`
- Target control: `reply-thread-menu`

What changed:
- Added a specific accessible name to the unresolved card's reply-thread menu button.
- Scoped the mobile recovery layout so the unresolved card title wraps onto its own line instead of clipping under the action cluster.
- Added the reopened mobile coverage entry for the hotfix route at `412x915`.

Rollback avoidance:
- Did not reapply the rolled-back broad `More actions` label pattern from `INC-4481`.
- Did not rely on hidden overflow or truncation to mask the layout bug.
