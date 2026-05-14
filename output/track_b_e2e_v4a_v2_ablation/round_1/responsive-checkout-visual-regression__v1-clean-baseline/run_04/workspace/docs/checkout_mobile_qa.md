# Checkout Mobile QA Note

## Issue Fixed
Mobile viewport (390x844) had compact summary and bottom bar overlapping the payment form, partially hiding the `Place order` CTA.

## Changes Made
1. **Removed negative margin**: Changed `.checkout-summary` `margin-bottom: -64px` to `margin-bottom: 0` to prevent overlap
2. **Added bottom padding**: Changed `.checkout-shell` padding to `16px 16px 100px` to reserve space for fixed mobile bar
3. **Fixed z-index stacking**: Added `z-index: 3` to `.checkout-mobile-bar` and `z-index: 2` to `.checkout-summary` for proper layering

## Verification
- Desktop two-column layout preserved (grid with 320px sidebar)
- Sticky summary behavior maintained on both desktop and mobile
- Compact summary experiment flag remains enabled (`data-preview-compact="true"`)
- Mobile bar now sits above content without obscuring CTA
