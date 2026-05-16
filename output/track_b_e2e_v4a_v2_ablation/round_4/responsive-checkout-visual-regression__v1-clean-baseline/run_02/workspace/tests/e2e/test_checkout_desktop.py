from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_viewport_has_bottom_padding_for_cta():
    """Mobile viewport assertion: payment form has padding to prevent overlap with bottom bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "padding-bottom: 80px" in css
    assert "position: fixed" in css
    assert ".checkout-mobile-bar" in css
