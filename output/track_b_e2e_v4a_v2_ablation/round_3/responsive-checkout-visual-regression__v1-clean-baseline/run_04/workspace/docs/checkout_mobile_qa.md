# Checkout Mobile QA Notes

## Issue Fixed
Mobile 390x844 viewport: compact summary and bottom bar overlapped the payment form, partially hiding the `Place order` CTA.

## Changes Made
- Removed negative `margin-bottom: -64px` from `.checkout-summary` (was causing overlap)
- Added `padding-bottom: 80px` to `.checkout-payment` to create space for fixed bottom bar
- Kept `position: sticky` on summary (required for compact experiment)
- Desktop two-column grid layout unchanged

## Test Coverage
Added `tests/e2e/test_checkout_mobile.py` with mobile viewport assertions:
- Verifies no negative margin overlap
- Confirms payment form has bottom padding
- Validates fixed bottom bar positioning
- Ensures CTA minimum height for accessibility height
