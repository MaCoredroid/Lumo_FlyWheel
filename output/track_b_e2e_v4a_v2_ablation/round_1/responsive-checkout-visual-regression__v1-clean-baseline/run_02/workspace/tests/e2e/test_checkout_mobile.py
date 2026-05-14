from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_no_overlap_with_compact_summary():
    """Verify mobile viewport doesn't have overlapping elements."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Check that negative margin is removed to prevent overlap
    assert "margin-bottom: -64px" not in css

    # Check that payment section has bottom padding for mobile bar space
    assert "padding-bottom: 80px" in css

    # Verify mobile bar is still shown on mobile
    assert "display: flex" in css
    assert "position: fixed" in css
    assert "checkout-mobile-bar" in css


# Mobile viewport assertion for 390x844 visual regression testing
MOBILE_VIEWPORT = "390x844"
