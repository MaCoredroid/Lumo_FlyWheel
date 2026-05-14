# Checkout Mobile QA Notes

## Issue
On mobile (390x844), the compact summary and bottom bar were overlapping the payment form, partially hiding the `Place order` CTA.

## Fix
- Removed `margin-bottom: -64px` from `.checkout-summary` on mobile (was causing negative space overlap)
- Added `padding-bottom: 80px` to `.checkout-payment` on mobile to ensure CTA clears the fixed bottom bar

## Testing
- Desktop two-column layout remains intact (grid with 320px summary column)
- Mobile now has proper spacing; CTA is fully visible and tappable
- Sticky summary behavior preserved on both desktop and mobile
- Compact summary experiment remains enabled

## Test Coverage
Added mobile viewport assertion in `tests/e2e/test_checkout_desktop.py::test_mobile_checkout_has_positive_padding_bottom_on_mobile()`
