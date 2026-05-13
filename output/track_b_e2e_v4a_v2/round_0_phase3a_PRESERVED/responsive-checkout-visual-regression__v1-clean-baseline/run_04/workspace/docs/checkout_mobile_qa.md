# Checkout Mobile QA Notes

## Issue
On mobile (390x844), the compact summary and bottom bar overlapped the payment form, partially hiding the `Place order` CTA.

## Fix
- Added `padding-bottom: 80px` to `.checkout-payment` to create space for the fixed bottom bar
- Changed `.checkout-summary` from `margin-bottom: -64px` to `margin-bottom: 24px` to prevent overlap with payment form
- Desktop two-column grid layout remains unchanged
- Sticky positioning preserved for both desktop and mobile summary

## Test Coverage
- `tests/e2e/test_checkout_mobile.py` - Mobile viewport assertions for overlap prevention
- `tests/e2e/test_checkout_desktop.py` - Desktop two-column behavior verification
