# Checkout Mobile QA Notes

## Issue
Compact summary and bottom bar overlapped the payment form on mobile (390x844), partially hiding the `Place order` CTA.

## Fix
- Changed `.checkout-summary` `margin-bottom` from `-64px` to `16px` to prevent overlap.
- Added `padding-bottom: 80px` to `.checkout-payment` to ensure the primary CTA has clearance from the fixed bottom bar.
- Desktop two-column layout and sticky summary behavior remain intact.

## Test Coverage
Added `test_mobile_viewport_has_bottom_padding_for_cta` in `tests/e2e/test_checkout_desktop.py` to assert mobile viewport styles.
