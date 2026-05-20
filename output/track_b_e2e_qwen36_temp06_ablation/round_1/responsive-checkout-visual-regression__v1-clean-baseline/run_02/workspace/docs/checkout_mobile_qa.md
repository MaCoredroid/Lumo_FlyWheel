# Checkout Mobile QA Note

## Issue
On mobile (390×844), the compact summary and fixed bottom bar overlapped the payment form, partially hiding the `Place order` CTA.

## Root Cause
- `.checkout-summary` had `margin-bottom: -64px`, pulling content into the payment form.
- `.checkout-mobile-bar` was `position: fixed` at the bottom with no compensating bottom padding.

## Verification
1. Open checkout at 390×844 viewport.
2. Confirm `Place order` button is fully visible and tappable.
3. Verify order summary sticks to top of viewport.
4. Desktop 1440px: two-column layout unchanged, sticky summary intact.
