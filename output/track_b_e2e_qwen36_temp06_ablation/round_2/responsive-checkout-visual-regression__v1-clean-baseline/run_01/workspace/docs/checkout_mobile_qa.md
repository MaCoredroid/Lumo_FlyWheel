# Checkout Mobile QA Note

## Issue
On mobile (390x844), the compact summary and fixed bottom bar overlapped the payment form, partially hiding the `Place order` CTA.

## Fix (`apps/storefront/styles/checkout.css`)
- Removed `margin-bottom: -64px` from the compact summary — this negative margin was pulling the summary into the payment form.
- Added `padding-bottom: 112px` to `.checkout-payment` on mobile so the CTA clears the fixed bottom bar.
- Added `margin-top: 8px` and `margin-bottom: 16px` on the compact summary for proper spacing.
- Added `border to the compact summary for visual separation.

## What was preserved
- Desktop two-column grid layout (unchanged).
- Sticky behavior on `.checkout-summary` (kept on both desktop and mobile).
- Compact summary experiment flag (`data-preview-compact="true"`) — still active.

## Verification
- Desktop 1440px: two-column layout with sticky summary — no change.
- Mobile 390x844: payment form scrolls freely, CTA is fully visible above the fixed bar, summary no longer overlaps form fields.
