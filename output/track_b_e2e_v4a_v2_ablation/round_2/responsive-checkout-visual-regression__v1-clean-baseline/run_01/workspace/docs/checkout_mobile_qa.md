# Checkout Mobile QA Note

## Issue
On mobile (390x844), the compact summary and bottom bar overlapped the payment form, partially hiding the `Place order` CTA.

## Fix
- Removed `margin-bottom: -64px` from `.checkout-summary` in mobile compact mode
- Added `padding-bottom: 100px` to `.checkout-payment` to account for fixed bottom bar

## Testing
- Desktop 1440px: Two-column layout preserved
- Mobile 390x844: No overlap; CTA fully visible and tappable
