# Checkout Mobile QA Notes

## Issue Fixed
Mobile 390x844 viewport: compact summary and bottom bar were overlapping the payment form, partially hiding the primary `Place order` CTA.

## Changes Made
1. Changed `.checkout-summary` `margin-bottom` from `-64px` to `16px` to eliminate overlap
2. Changed `.checkout-mobile-bar` from `position: fixed` to `position: sticky` for proper document flow
3. Added `padding-bottom: 80px` to `.checkout-payment` to ensure CTA has space above the sticky bar

## Desktop Behavior
- Two-column grid layout preserved at widths > 600px
- Sticky summary sidebar unchanged

## Test Coverage
Added `test_mobile_viewport_no_overlap()` in `tests/e2e/test_checkout_desktop.py` to verify:
- No negative margin values causing overlap
- Payment section has adequate bottom padding
- Mobile bar uses sticky positioning
