from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_compact_no_negative_margin():
    """Mobile compact summary must not use negative margin that causes overlap."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "margin-bottom: -64px" not in css
    assert "margin-bottom: 0" in css


def test_mobile_compact_shell_has_bottom_padding_for_fixed_bar():
    """Mobile compact shell must have bottom padding to clear the fixed bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "padding: 16px 16px 88px" in css
