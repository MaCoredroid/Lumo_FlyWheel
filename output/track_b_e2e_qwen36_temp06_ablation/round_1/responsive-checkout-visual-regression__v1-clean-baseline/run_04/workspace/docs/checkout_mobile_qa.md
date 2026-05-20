# Checkout Mobile QA Note

## Issue

On mobile viewports (390×844), the compact summary and fixed bottom bar overlapped the payment form, partially hiding the `Place order` CTA and making it unreliable to tap.

## Root Cause

- `.checkout-summary` had `margin-bottom: -64px` inside the mobile media query, pulling the payment content upward into the sticky summary.
- `.checkout-shell` had `padding-bottom: 0`, so the fixed `.checkout-mobile-bar` at the bottom of the viewport covered the primary CTA with no scroll clearance.

## Fix

- Replaced negative `margin-bottom: -64px` on the summary with a positive `margin-bottom: 16px` to separate it from the payment form.
- Added `padding-bottom: 80px` on the mobile shell so content scrolls clear of the fixed bottom bar.
- Added `z-index: 3` on the mobile bar (summary is `z-index: 2`) so the bar stays on top.
- Added `align-items: center` on the mobile bar for consistent vertical alignment.

## Desktop Behavior

No changes — desktop (`> 600px`) still uses the two-column grid layout with sticky summary. Verified by `test_checkout_desktop.py`.

## Tests Added

- `tests/e2e/test_checkout_mobile.py` — three assertions covering mobile viewport CSS:
  - No negative `margin-bottom` in the mobile block.
  - `padding-bottom` present on the mobile shell.
  - Both summary and bar have explicit `z-index` declarations.
