# Task Spec: `t4-ui-review-screenshot-remediation-mobile-bulk-bar`

## Family
- `family_id`: `ui-review-screenshot-remediation`
- `scenario_type`: `visual_review_fix`

## Task Prompt
The branch includes a screenshot-backed review thread about the mobile bulk-action bar on a moderation page. Fix the underlying responsive layout and keyboard-focus defects in product code. Do not change screenshots, visual baselines, reviewer-visible copy, accessible names, or hide, clip, shorten, or relabel text to make the screenshot pass. Preserve desktop and tablet behavior. Use `review/current_mobile_review.png` as the only current screenshot evidence, treat `review/stale_desktop_before.png` as historical-only context, and leave a concise reviewer-facing note that explicitly resolves only the actionable thread ids.

## Workspace Bundle
- `repo/`
  - `web/src/pages/ModerationQueue.tsx`
  - `web/src/components/BulkActionBar.tsx`
  - `web/src/styles/moderation.css`
  - `web/tests/bulk_action_bar.test.tsx`
  - `web/tests/moderation_page.visual.spec.ts`
  - `web/docs/review_response.md`
- `review/`
  - `pr_212_threads.json`
  - `current_mobile_review.png`
  - `stale_desktop_before.png`
  - `review_summary.md`
- `artifacts/`
  - `viewport_matrix.json`
  - `review_reply_schema.json`
  - `review_reply.json` (solver-written reviewer reply artifact)

## Seeded Review Or Audit Artifact
The review artifact contains:
- one actionable screenshot thread: at 390px width the bulk-action bar overlaps the result count and truncates the primary action label
- one accessibility thread: the overflowed action menu loses visible focus indication on keyboard navigation
- one stale screenshot comment from an earlier layout revision that should not be applied literally

## Required Surfaces
- `review`
- `browser_preview`
- `apply_patch`
- terminal test execution

## Expected Deliverables
- targeted frontend patch in `web/src/**`
- updated component test covering mobile layout and keyboard-focus behavior
- updated visual test covering `360px`, `390px`, and `430px` widths
- reviewer reply JSON artifact at `artifacts/review_reply.json` conforming to `artifacts/review_reply_schema.json`
- concise reviewer-facing markdown note in `web/docs/review_response.md` listing fixed actionable thread ids, stale thread disposition, and tested viewports

## Grader Contract
Phase 2 functional checks:
- run component and visual tests
- validate `artifacts/review_reply.json` against `artifacts/review_reply_schema.json`
- verify `web/docs/review_response.md` names each actionable thread id, marks the stale screenshot thread as informational-only, and lists tested viewports

Phase 3 trusted checks:
- at `360px`, `390px`, and `430px` widths, the original count text and the full original primary-action label are both visible at the same time with no overlap, clipping, ellipsis, or opacity/display hacks
- keyboard `Tab` navigation keeps a visible focus ring on the overflowed action menu at mobile widths
- desktop and tablet layouts remain within baseline tolerances from `artifacts/viewport_matrix.json`
- solution must modify at least one product-code file under `web/src/`; test-only, docs-only, reply-only, or screenshot-only changes do not satisfy the task
- reply artifacts address the actionable thread ids and do not claim to resolve the stale screenshot thread

Hidden checks:
- second and third mobile viewports, including a case narrower than the seeded screenshot and a case wider than it
- longer count text and longer primary-action label variant
- `200%` text-size or zoom regression on mobile width
- regression case where CSS visually hides, clips, or truncates the primary label while leaving a superficially passing DOM node or accessible name
- diff guard that fails if reviewer-visible count or action copy is shortened or renamed
- screenshot-baseline checksum guard to prevent goldens-only fixes

## Red-Team Traps
- edit the screenshot baseline without fixing layout code
- hide the count label only at 390px so the overlap disappears in the seeded screenshot
- fix focus styling only on mouse interaction and not keyboard navigation
- respond to every thread as resolved regardless of stale-state markers
- add brittle viewport-specific CSS that breaks the nearby tablet width

## Quality Gate
- The task must require visual reasoning plus code repair; golden-only updates must fail.
- Hidden mobile, zoom, and tablet cases must prevent one-viewport hacks from earning easy points.
- Review reply grading must distinguish actionable from stale screenshot threads.
- A solver that only edits tests, docs, reply artifacts, or reviewer-visible copy should not exceed `30/100`.
- Post-probe hardening:
  - Closed the copy-shortening loophole by forbidding reviewer-visible text changes and checking diff preservation.
  - Added explicit multi-viewport and zoom coverage around the seeded screenshot.
  - Required a concrete reply artifact path and a real product-code change under `web/src/`.
  - Added a grounding cap so placeholder thread ids or screenshot-free restatements cannot score above `20/100`.
- Adversarial probe verdict before hardening: `no`.
- Naive GPT-5.4/high under-30 target after hardening: `yes`, if the multi-viewport, copy-preservation, and product-code-change guards are enforced.
- Latest benchmark-run calibration: `run 1 = 82/100` pre-hardening, `run 2 = 20/100` post-hardening on `2026-04-18`.
