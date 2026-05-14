# Checkout Mobile QA Notes

## Issue Fixed
Mobile viewport (390x844) had overlapping elements when compact summary was enabled:
- Order summary overlapped the payment form
- Bottom mobile bar partially hid the "Place order" CTA

## Changes Made

### CSS (`apps/storefront/styles/checkout.css`)
1. Changed `margin-bottom: -64px` to `margin-bottom: 16px` on `.checkout-summary` to prevent overlap with payment form
2. Added `padding-bottom: 80px` to `.checkout-payment` to create space for fixed bottom bar

## Verification
- Desktop 1440px: Two-column layout remains intact with sticky summary
- Mobile 390x844: No overlap; primary CTA fully visible and tappable
- Compact summary experiment remains enabled
- Sticky behavior preserved for desktop

## Test Coverage
Added `tests/e2e/test_checkout_mobile.py` with assertions for:
- No negative margin causing overlap
- Bottom padding for mobile bar clearance
- Mobile bar visibility and positioning
