# Checkout Mobile QA Note

## Problem

On mobile 390×844 viewports, the compact summary and fixed bottom bar overlapped the payment form, partially hiding the Place order CTA.

## Root cause

- `.checkout-summary` had `margin-bottom: -64px` on mobile, pulling the sticky summary over the payment form.
- `.checkout-shell` had no bottom padding, so the fixed `.checkout-mobile-bar` covered the CTA button.

## Fix

- Replaced `margin-bottom: -64px` with `margin-bottom: 16px` on the compact summary — summary no longer overlaps payment content.
- Added `padding-bottom: 80px` on `.checkout-shell` in the mobile media query — ensures the CTA sits above the fixed bottom bar.
- Desktop two-column grid and sticky summary behavior are unchanged.

## Test coverage

- `tests/e2e/test_checkout_mobile.py` — asserts the negative margin is gone and bottom padding is present.
- `tests/e2e/test_checkout_desktop.py` — unchanged, still asserts two-column grid and sticky summary.
- `tests/unit/test_checkout_copy.py` — unchanged, still asserts CTA copy.

## Manual verification

1. Open checkout at 390×844 viewport.
2. Confirm the order summary sits above the payment form without overlap.
3. Scroll to the bottom and confirm the Place order button is fully visible and tappable.
4. Verify desktop 1440px still shows the two-column layout with sticky summary.
