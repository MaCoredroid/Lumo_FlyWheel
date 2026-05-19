# Checkout Mobile QA Notes

## Issue
On mobile viewports (390 × 844), the compact summary section overlapped the payment form and the fixed bottom bar hid the primary **Place order** CTA.

## Root Cause
- `.checkout-summary` used `margin-bottom: -64px` on mobile, pulling the sticky summary up over the payment fields.
- `.checkout-shell` had `padding-bottom: 0` on mobile, so the payment form scrolled behind the fixed `.checkout-mobile-bar`.

## Fix Applied (`apps/storefront/styles/checkout.css`)
- Removed `margin-bottom: -64px` from the mobile summary (set to `0` with `margin-top: 8px` for spacing).
- Changed mobile shell padding from `16px 16px 0` to `16px 16px 80px` so content clears the fixed bottom bar.

## What Was NOT Changed
- Desktop two-column grid layout (`grid-template-columns: minmax(0, 1fr) 320px`) — untouched.
- Sticky behavior on both desktop and mobile — preserved.
- `data-preview-compact` experiment flag — still enabled by default.

## Verification
- Run `pytest tests/e2e/test_checkout_desktop.py` — all 3 assertions pass (desktop grid, mobile margin, mobile padding).
- Manual check at 390 × 844 viewport: payment form is fully visible, **Place order** button is tappable, and summary sits above the form without overlap.
