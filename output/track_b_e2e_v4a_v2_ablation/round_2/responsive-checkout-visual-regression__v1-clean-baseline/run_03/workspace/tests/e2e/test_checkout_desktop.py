from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_viewport_assertions():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Desktop two-column layout
    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert "position: sticky" in css

    # Mobile viewport: compact summary with fixed bottom bar
    assert "@media (max-width: 600px)" in css
    assert ".checkout-mobile-bar" in css
    assert "position: fixed" in css
    assert "bottom: 0" in css

    # Mobile: padding-bottom on shell to prevent content overlap with fixed bar
    assert "padding: 16px 16px 100px" in css

    # Mobile: summary has positive margin-bottom (not negative) to prevent overlap
    assert "margin-bottom: 24px" in css
