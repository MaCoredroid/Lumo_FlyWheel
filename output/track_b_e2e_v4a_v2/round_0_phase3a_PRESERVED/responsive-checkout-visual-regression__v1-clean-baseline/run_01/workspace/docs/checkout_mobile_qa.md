# Checkout Mobile QA Notes

## Issue
On mobile (390x844), the compact summary and fixed bottom bar overlapped the payment form, partially hiding the `Place order` CTA.

## Fix
Added `padding-bottom: 80px` to `.checkout-payment` in mobile compact mode to prevent overlap with the fixed bottom bar.

## Verification
- Desktop two-column layout: unchanged (grid with 320px summary column)
- Mobile compact summary: sticky at top with negative margin preserved
- Mobile bottom bar: fixed at bottom, no longer overlaps CTA
- Primary CTA: fully visible and tappable

## Test Coverage
- `tests/e2e/test_checkout_mobile.py`: asserts mobile padding and CTA visibility
- `tests/e2e/test_checkout_desktop.py`: asserts desktop two-column behavior
