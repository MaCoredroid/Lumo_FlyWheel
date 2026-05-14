from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_checkout_viewport_assertions():
    """Mobile viewport (390x844) assertions for checkout layout."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile bar space padding
    assert "padding: 16px 16px 100px" in css

    # No negative margin causing overlap
    assert "margin-bottom: -64px" not in css

    # Mobile bar z-index above content
    assert "z-index: 3" in css

    # Summary z-index below mobile bar
    assert "z-index: 2" in css
