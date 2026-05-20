# Checkout Mobile QA Notes
## Checkout Mobile — QA Notes

**Viewport tested:** 390 × 844 (mobile)

**Problem**
- The compact summary (`margin-bottom: -64px`) overlapped the payment form.
- The fixed bottom bar covered the *Place order* CTA, making it unreliable to tap.

**Fix**
- Removed the negative margin on `.checkout-summary` (now `margin-bottom: 16px`).
- Added `padding-bottom: 112px` to the shell so scrollable content clears the fixed bar.
- Set `z-index: 10` on `.checkout-mobile-bar` so it sits above page content.
- Desktop two-column grid and sticky summary are unchanged.

**Verification**
- Run `pytest tests/e2e/test_checkout_mobile.py` — 3 assertions pass.
- Run `pytest tests/e2e/test_checkout_desktop.py` — desktop grid + sticky still asserted.
