# Checkout Mobile QA Notes

## Issue Fixed
On mobile (390x844 viewport), the compact summary and bottom bar were overlapping the payment form, partially hiding the "Place order" CTA.

## Root Cause
- `.checkout-summary` had `margin-bottom: -64px` pulling content upward
- No bottom padding on `.checkout-payment` to account for fixed bottom bar

## Solution
- Removed negative margin from `.checkout-summary` (changed to `margin-bottom: 0`)
- Added `padding-bottom: 80px` to `.checkout-payment` in mobile breakpoint, the payment section now has adequate spacing to prevent overlap.

## Test Coverage
Added `test_mobile_checkout_prevents_overlap_with_padding()` assertion to `tests/e2e/test_checkout_desktop.py`.

## Desktop Behavior
Two-column grid layout remains intact with sticky summary sidebar.
