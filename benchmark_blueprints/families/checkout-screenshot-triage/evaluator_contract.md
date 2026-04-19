# Evaluator Contract: Checkout Screenshot Triage

## Purpose
Score whether a solver repaired the real mobile checkout regression and its accessibility failure instead of overfitting a single screenshot, a single viewport, or the visible Playwright test.

## 100-Point Breakdown
- `25` Root-cause repair across the affected mobile checkout state.
- `20` Robust mobile layout behavior across viewport, safe-area, and content variations.
- `15` Correct post-validation focus target and accessibility behavior.
- `15` Desktop and sticky-summary non-regression.
- `10` Fresh machine-generated evidence from the repaired flow.
- `10` Test integrity and anti-shortcut discipline.
- `5` Release note quality.

## Scoring Caps
- Cap at `25/100` if the submission has no fresh machine-generated screenshots, traces, or equivalent runtime evidence.
- Cap at `20/100` if the submission does not change runtime checkout behavior and only offers reasoning, test edits, or screenshot edits.
- Cap at `35/100` if the submission fixes layout but leaves focus semantics ambiguous or incorrect.
- Cap at `10/100` if the submission changes tests, timing, viewport setup, or baselines without a runtime fix.

## Visible Pass Signals
- The visible coupon-checkout test passes.
- Desktop smoke behavior is preserved.
- Submission includes a short release note and fresh evidence.

## Hidden Evaluation Signals
- Hidden checks vary viewport height, safe-area inset, saved-address copy, and coupon label length.
- Hidden checks verify the CTA remains visible before and after validation failure.
- Hidden checks require focus to land on a visible actionable error target or the first invalid control.
- Hidden checks reject disabled sticky behavior, clipped content, and test-only fixes.

## Shortcut Failure Modes
- Fixed offset or padding passes one coupon fixture only.
- Focus moves to a hidden wrapper or unchanged submit button.
- Coupon rendering is suppressed to avoid overflow.
- Evidence is stale, cropped, or manually substituted.

## Calibration Target
- Design-time family-package run target: `15-25/100`.
- Family freeze target for a naive GPT-5.4/high solver with the full benchmark bundle: roughly `20/100`, and not above `25/100`.

