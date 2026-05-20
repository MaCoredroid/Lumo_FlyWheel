# Checkout Mobile QA Notes

## Problem
On mobile (390x844) with the compact summary experiment enabled, the sticky order summary and the fixed bottom bar overlapped the payment form, partially hiding the **Place order** CTA.

## Fix Applied
- **Removed negative margin** - `.checkout-summary` `margin-bottom` changed from `-64px` to `16px` so the summary no longer overlaps the payment form.
- **Added bottom padding** - `.checkout-shell` padding changed from `16px 16px 0` to `16px 16px 80px` so the payment form scrolls past the fixed bottom bar, keeping the CTA tappable.

## What Was NOT changed
- Compact summary experiment (`data-preview-compact`) remains enabled.
- Sticky behavior on `.checkout-summary` is preserved.
- Desktop two-column grid layout is untouched.

## Manual verification steps
1. Open checkout at 390x844 viewport.
2. Confirm the order summary sits above the payment form without overlap.
3. Scroll to the bottom and confirm **Place order** CTA is fully visible and tappable.
4. Confirm the fixed bottom bar shows total and **Review order** button.
5. Verify desktop (>=601px) still renders the two-column grid with sticky summary.
