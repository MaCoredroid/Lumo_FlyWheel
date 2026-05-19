# Checkout Mobile QA Note

## Problem

On mobile viewports (390×844), the compact summary sidebar and the fixed bottom bar overlapped the payment form, partially hiding the **Place order** CTA.

## Root Cause

- `.checkout-summary` had `margin-bottom: -64px`, pulling the payment section upward into the sticky summary.
- `.checkout-shell` had `padding-bottom: 0`, so the payment form extended behind the fixed `.checkout-mobile-bar`.
- No explicit `z-index` on the mobile bar, so stacking order was unreliable.

## Fix (in `apps/storefront/styles/checkout.css`)

| Change | Before | After |
|--------|--------|-------|
| Shell bottom padding | `16px 16px 0` | `16px 16px 80px` — clears the ~64 px fixed bar |
| Summary margin-bottom | `-64px` | `16px` — positive spacing, no overlap |
| Mobile bar z-index | (none) | `z-index: 3` — above sticky summary (`z-index: 2`) |
| Mobile bar alignment | (none) | `align-items: center` — vertical centering |

## What Was Preserved

- **Desktop two-column layout** — grid `grid-template-columns: minmax(0, 1fr) 320px` unchanged.
- **Sticky summary** — `position: sticky` retained on both desktop and mobile.
- **Compact summary experiment** — `data-preview-compact="true"` behavior intact.

## Manual Verification

1. Open checkout at **390×844** viewport.
2. Confirm the **Place order** button is fully visible and tappable.
3. Confirm the order summary sits above the payment form without overlap.
4. Confirm the fixed bottom bar stays visible during scroll.
5. Open at **1440×900** — verify two-column layout is unchanged.
