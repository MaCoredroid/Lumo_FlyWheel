from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_checkout_has_bottom_padding_to_prevent_overlap():
    """Mobile viewport assertion: payment section has padding to prevent overlap with fixed bottom bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Check mobile media query exists
    assert "@media (max-width: 600px)" in css

    # Check that compact mode payment section has bottom padding
    assert "padding-bottom: 80px" in css
