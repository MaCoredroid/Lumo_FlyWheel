# Checkout Mobile QA Notes

## Issue
On mobile (390x844), the compact summary and bottom bar overlapped elements hid the `Place order` CTA when compact summary was enabled.

## Fix
- Removed negative margin on `.checkout-summary` and added `padding-bottom: 100px` to `.checkout-payment` to prevent content overlap with the fixed bottom bar.

## Test
- `tests/e2e/test_checkout_mobile.py` includes mobile viewport assertions for compact summary mode.
- Desktop two-column layout preserved (verified by `test_checkout_desktop.py`).
