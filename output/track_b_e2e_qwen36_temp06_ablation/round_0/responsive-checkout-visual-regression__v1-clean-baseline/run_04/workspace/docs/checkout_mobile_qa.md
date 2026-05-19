# Checkout Mobile QA Note

## Issue
On mobile (390 × 844), the compact summary and fixed bottom bar overlapped the payment form, partially hiding the **Place order** CTA and making it unreliable to tap.

## Root Cause
- `.checkout-summary` used `margin-bottom: -64px` on mobile, pulling the summary upward into the payment form.
- `.checkout-shell` had `padding-bottom: 0`, so the fixed `.checkout-mobile-bar` covered the bottom of the payment section including the primary CTA.

## Fix (apps/storefront/styles/checkout.css)
- Removed the negative `margin-bottom` on the mobile compact summary (now `16px` for normal spacing).
- Added `padding-bottom: 80px` on `.checkout-shell` so content clears the fixed bottom bar.
- Added `padding-bottom: 16px` on `.checkout-payment` for extra CTA clearance.
- Added `z-index: 10` on `.checkout-mobile-bar` to ensure it stays on top.

## What Was NOT Changed
- Desktop two-column grid layout (`grid-template-columns: minmax(0, 1fr) 320px`) — untouched.
- Sticky behavior on `.checkout-summary` — preserved on both desktop and mobile.
- Compact summary experiment (`data-preview-compact="true"`) — still enabled.

## Manual Verification Steps
1. Open checkout at **390 × 844** viewport.
2. Confirm the order summary sits above the payment form without overlap.
3. Scroll to the bottom — confirm **Place order** button is fully visible and tappable.
4. Confirm the fixed bottom bar shows total and "Review order".
5. Open at **1440px** — confirm two-column layout is unchanged.

## Automated Tests Added
- `tests/unit/test_checkout_mobile.py` — asserts no negative margin on mobile summary, sufficient bottom padding on shell, and padding-bottom on payment section.
