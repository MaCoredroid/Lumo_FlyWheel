# Checkout Mobile QA

## Issue Fixed
On mobile (390x844), the compact summary and bottom bar were overlapping the payment form, partially hiding the `Place order` CTA.

## Changes Made

### CSS (`apps/storefront/styles/checkout.css`)
1. Changed `padding: 16px 16px 0` to `padding: 16px 16px 100px` on `.checkout-shell[data-preview-compact="true"]` to add bottom padding equal to the fixed bar height, preventing content overlap.
2. Changed `margin-bottom: -64px` to `margin-bottom: 24px` on the summary to remove the negative margin that caused overlap.
3. Added `z-index: 3` to `.checkout-mobile-bar` to ensure proper stacking.

## Test Updates
Added `test_mobile_viewport_assertions` in `tests/e2e/test_checkout_desktop.py` to verify:
- Mobile media query exists
- Fixed bottom bar positioning
- Proper padding to prevent overlap
- Positive margin on summary

## Desktop Behavior Preserved
- Two-column grid layout intact (`grid-template-columns: minmax(0, 1fr) 320px`)
- Sticky summary behavior maintained
- Compact summary experiment flag still functional
