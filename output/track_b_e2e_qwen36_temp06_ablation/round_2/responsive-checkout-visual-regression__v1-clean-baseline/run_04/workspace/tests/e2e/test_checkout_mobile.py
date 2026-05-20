from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_no_overlap():
    """Mobile viewport: compact summary must not overlap the payment form,
    and the CTA must remain reachable (bottom padding on payment section)."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile media query must exist
    assert "@media (max-width: 600px)" in css

    # The payment section needs bottom padding so the fixed bar doesn't hide the CTA
    assert "padding-bottom" in css

    # The compact summary must NOT use negative margin (that caused the overlap)
    assert "margin-bottom: -" not in css
