# Evaluator Contract: Responsive Checkout Visual Regression

## Scoring Breakdown

Total raw score: 100 points.

1. Root-cause investigation quality: 15
   - Identifies mobile-only preview regression rather than generic checkout failure: 5
   - Distinguishes CTA occlusion from unrelated console noise: 5
   - Uses viewport-specific observations rather than spec paraphrase alone: 5
2. Concrete workspace edits: 20
   - Names likely component/style/test targets with file-level specificity: 10
   - Proposes a responsive fix that preserves the preview experiment path instead of disabling it: 10
3. Preview/browser evidence: 20
   - 390x844 clickability and visibility evidence: 8
   - Hidden second-mobile viewport evidence: 6
   - Tablet/desktop preservation evidence: 6
4. Regression and anti-shortcut handling: 15
   - Rejects experiment disablement or global sticky removal: 5
   - Guards against screenshot-only or one-breakpoint hacks: 5
   - Preserves totals and compact summary presence: 5
5. Deliverables and docs: 10
   - QA note with tested viewports: 5
   - Before/after evidence bundle accounted for: 5
6. Family-specific correctness invariants: 20
   - Mobile CTA is visible: 5
   - Mobile CTA is clickable and not occluded: 5
   - Compact summary remains active under the preview flag: 5
   - Tablet and desktop layout remain stable: 5

## Global Caps

- No concrete file-level patch or diff-backed change set: total capped at 45.
- No trusted preview-browser viewport evidence: total capped at 25.
- Response relies only on task text and gives no geometry, hit-testing, or viewport-specific observations: total capped at 20.
- Any banned shortcut, including experiment disablement, global sticky removal, or screenshot-only fixes: total capped at 10.

## Automatic Zero Conditions

- The proposal hides the CTA issue by disabling the preview experiment.
- The proposal removes sticky behavior entirely across all viewports.
- The proposal supplies only screenshot edits or docs edits without a layout change strategy.

## Scoring Notes

- A family-local artifact-only attempt should usually land between 15 and 25 because it cannot produce real viewport evidence or code diffs.
- The task is correctly hardened when a naive GPT-5.4/high child attempt from this directory stays near 20/100.

## Final Hardness Judgment

- Target naive GPT-5.4/high score: approximately 20/100.
- Current judgment: meaningful and sufficiently hardened if the latest `benchmark_run.md` score is within 15-25 and limited by the viewport-evidence cap.
