# Checkout Mobile QA Note

## Problem

On mobile (390×844), the compact order summary and the fixed bottom bar overlapped the payment form, partially hiding the **Place order** CTA.

## Root Cause

- `.checkout-summary` used `margin-bottom: -64px`, pulling the summary into the payment form.
- `.checkout-payment` had no bottom padding, so the fixed `.checkout-mobile-bar` covered the CTA.
- Shell padding-bottom was `0`, leaving no scroll space below the form.

## Fix (apps/storefront/styles/checkout.css)

| Change | Before | After |
|---|---|---|
| Shell padding | `16px 16px 0` | `16px` |
| Summary margin | `margin-bottom: -64px` | `margin-bottom: 16px` |
| Mobile bar | — | `min-height: 64px` |
| Payment section | — | `padding-bottom: 80px` |

## What Was Preserved

- Desktop two-column grid layout (`minmax(0, 1fr) 320px`) — unchanged.
- Sticky summary behavior on desktop — unchanged.
- Compact summary experiment (`data-preview-compact="true"`) — still enabled.

## Verification

- `tests/e2e/test_checkout_desktop.py` — confirms desktop grid + sticky summary intact.
- `tests/e2e/test_checkout_mobile.py` — asserts no negative margin and payment bottom padding present.
- `tests/unit/test_checkout_copy.py` — confirms CTA copy unchanged.
