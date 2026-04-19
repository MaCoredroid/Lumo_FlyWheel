# Screenshot Review Fix

Use this skill when solving the `ui-review-screenshot-remediation` family.

## Workflow
1. Treat the current screenshot as active evidence and stale screenshots as history only.
2. Fix the layout in product code, not in the screenshot baseline.
3. Preserve reviewer-visible text and accessible names.
4. Cover mobile widths `360px`, `390px`, and `430px` plus keyboard focus behavior.
5. Write a stale-aware reviewer response instead of resolving every thread.

## Avoid
- shortening labels or count text
- viewport-specific hacks at one width only
- mouse-only focus styling
- reply artifacts that mark stale feedback as fixed
