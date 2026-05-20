from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_compact_summary_no_overlap():
    """Mobile compact summary should not overlap the payment form or hide the CTA."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The summary must not use a negative margin that causes overlap
    assert "margin-bottom: -64px" not in css

    # The shell must have bottom padding to account for the fixed mobile bar
    assert "padding: 16px 16px 80px" in css

    # The mobile bar must be present and fixed at the bottom
    assert ".checkout-mobile-bar" in css
    assert "position: fixed" in css
    assert "bottom: 0" in css
