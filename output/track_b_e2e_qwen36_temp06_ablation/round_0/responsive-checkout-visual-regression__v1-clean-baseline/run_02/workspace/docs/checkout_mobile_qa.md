# Checkout Mobile QA Note

## Problem

On mobile viewports (390 × 844), the compact order summary and fixed bottom bar overlapped the payment form, partially hiding the **Place order** CTA and making it unreliable to tap.

## Root Cause

`apps/storefront/styles/checkout.css` (mobile media query):

- `.checkout-summary` had `margin-bottom: -64px`, pulling the sticky summary up into the payment form.
- `.checkout-shell` had `padding-bottom: 0`, leaving no clearance for the fixed `.checkout-mobile-bar` at the bottom of the viewport.

## Fix Applied

- Replaced `margin-bottom: -64px` with `margin-bottom: 16px` on the mobile compact summary so it no longer overlaps the payment form.
- Changed `.checkout-shell` padding from `16px 16px 0` to `16px 16px 80px` so the payment form (including the CTA) clears the fixed bottom bar.

## What Was Preserved

- Desktop two-column grid layout (`grid-template-columns: minmax(0, 1fr) 320px`) — unchanged.
- Sticky summary behavior (`position: sticky`) — retained on both desktop and mobile.
- Compact summary experiment (`data-preview-compact="true"`) — still enabled.

## Tests Added

`tests/e2e/test_checkout_mobile.py` — 3 assertions:

1. No negative `margin-bottom` on the mobile compact summary.
2. Mobile shell has `padding-bottom: 80px` for the fixed bar.
3. Mobile bar remains `position: fixed` at `bottom: 0`.

All existing tests (desktop + unit) continue to pass.
