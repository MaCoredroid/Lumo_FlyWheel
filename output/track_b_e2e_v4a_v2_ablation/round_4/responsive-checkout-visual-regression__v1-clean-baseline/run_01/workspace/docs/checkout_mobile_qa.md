# Checkout Mobile QA Notes

## Issue Fixed
Mobile viewport (390x844) had compact summary and bottom bar overlapping the payment form, partially hiding the "Place order" CTA.

## Root Cause
- `.checkout-summary` used `margin-bottom: -64px` pulling content into the fixed bottom bar area
- `.checkout-shell` lacked bottom padding to account for the fixed `.checkout-mobile-bar`
- No z-index layering defined for the mobile bar

## Fix Applied
1. Changed `.checkout-shell[data-preview-compact="true"]` padding from `16px 16px 0` to `16px 16px 100px` to create space for the fixed bottom bar
2. Changed `.checkout-summary` from `margin-bottom: -64px` to `margin-bottom: 0` to prevent content overlap
3. Added `z-index: 3` to `.checkout-mobile-bar` for proper stacking
4. Added `padding-bottom: 16px` to `.checkout-payment` for visual breathing room

## Preserved Behavior
- Desktop two-column grid layout intact
- Sticky summary behavior maintained on both desktop and mobile
- Compact summary experiment flag (`data-preview-compact`) still functional

## Test Coverage
Added `/tests/e2e/test_checkout_mobile.py` with assertions for:
- Mobile bottom padding presence
- Sticky summary without negative margin
- Mobile bar z-index layering
