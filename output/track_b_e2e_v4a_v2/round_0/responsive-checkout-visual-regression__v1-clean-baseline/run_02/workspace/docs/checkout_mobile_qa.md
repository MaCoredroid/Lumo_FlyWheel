# Checkout Mobile QA Notes

## Issue Fixed
Mobile 390x844 viewport: compact summary and bottom bar were overlapping the payment form, partially hiding the `Place order` CTA.

## Root cause was `margin-bottom: -64px` on `.checkout-summary` pulling content upward.

## Solution Applied
- Removed negative margin from `.checkout-summary`
- Added `padding-bottom: 72px` to `.checkout-payment` to create space for fixed bottom bar
- Set `z-index: 3` on `.checkout-mobile-bar` for proper stacking

## Verified
- Desktop two-column layout preserved (320px summary column)
- Compact summary experiment (`data-preview-compact="true"`) remains enabled
- Sticky behavior maintained on both desktop and mobile
- All tests passing including new mobile viewport assertion
