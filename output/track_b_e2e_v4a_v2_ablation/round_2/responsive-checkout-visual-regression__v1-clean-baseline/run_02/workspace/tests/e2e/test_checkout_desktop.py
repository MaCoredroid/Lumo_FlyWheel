from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_checkout_has_positive_padding_bottom_on_mobile():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile viewport assertion: payment section has padding-bottom to prevent overlap
    assert "padding-bottom: 80px" in css
    # Ensure negative margin is removed (was -64px causing overlap)
    assert "margin-bottom: -64px" not in css
    # Mobile bar should still be present
    assert ".checkout-mobile-bar" in css
