from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_checkout_has_bottom_padding_for_floating_bar():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "padding-bottom: 80px" in css
    assert "margin-bottom: 16px" in css
    assert ".checkout-shell[data-preview-compact=\"true\"] .checkout-payment" in css or ".checkout-payment" in css
