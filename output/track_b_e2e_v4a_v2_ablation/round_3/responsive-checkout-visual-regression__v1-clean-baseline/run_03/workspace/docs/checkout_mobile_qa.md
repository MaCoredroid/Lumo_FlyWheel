# Checkout Mobile QA Notes

## Issue Fixed
Mobile overlap between compact summary, payment form, and bottom bar on 390x844 viewport.

## Root Cause
The `.checkout-summary` element had `margin-bottom: -64px` to visually overlap with the payment form, but the payment section lacked corresponding bottom padding. This caused the primary "Place order" CTA to be partially hidden under the fixed `.checkout-mobile-bar`.

## Solution
Added `padding-bottom: 80px` to `.checkout-payment` within the mobile media query. This creates space below the payment form so the CTA remains fully visible and tappable above the fixed bottom bar.

## Testing
- Desktop (1440px): Two-column layout preserved with sticky summary
- Mobile (390x844): Payment form has adequate spacing; CTA fully visible and accessible
- Run tests: `pytest tests/e2e/test_checkout_desktop.py`
