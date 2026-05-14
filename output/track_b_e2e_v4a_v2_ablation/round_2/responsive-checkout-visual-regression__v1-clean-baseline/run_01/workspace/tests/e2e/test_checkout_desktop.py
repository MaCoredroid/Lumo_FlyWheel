from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css_path = ROOT / "apps/storefront/styles/checkout.css"
    css = css_path.read_text(encoding="utf-8")

    assert "grid-template" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_checkout_no_negative_margin_overlap():
    """Mobile compact summary should not use negative margin overlap."""
    css_path = ROOT / "apps/storefront/styles/checkout.css"
    css = css_path.read_text(encoding="utf-8")

    # The fix: removed negative margin-bottom that caused overlap
    assert "-64px" not in css

    # Mobile payment section should have padding for fixed bottom bar
    assert "padding-bottom: 100px" in css

    # Mobile bar should still be fixed at bottom
    assert "position: fixed" in css
    assert "bottom: 0" in css
