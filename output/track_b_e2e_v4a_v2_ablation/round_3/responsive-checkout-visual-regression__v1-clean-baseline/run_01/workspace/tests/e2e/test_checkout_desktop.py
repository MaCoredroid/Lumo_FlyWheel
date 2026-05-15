from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_checkout_has_bottom_padding_for_cta():
    """Verify mobile viewport has padding to prevent CTA overlap with fixed bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Check mobile media query exists
    assert "@media (max-width: 600px)" in css

    # Check payment section has bottom padding for CTA clearance
    assert "padding-bottom: 80px" in css

    # Check mobile bar has proper z-index to sit above content
    assert "z-index: 10" in css
