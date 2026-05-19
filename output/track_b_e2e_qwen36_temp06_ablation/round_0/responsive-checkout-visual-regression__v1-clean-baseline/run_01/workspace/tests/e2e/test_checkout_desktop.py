from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_compact_summary_no_negative_margin():
    """Mobile compact summary must not use negative margin-bottom that overlaps content."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The negative margin-bottom that caused the overlap must be gone
    assert "margin-bottom: -64px" not in css


def test_mobile_shell_has_bottom_padding_for_fixed_bar():
    """Mobile shell must have bottom padding so the CTA isn't hidden behind the fixed bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Inside the mobile media query, the shell needs bottom padding >= 80px
    # to account for the fixed checkout-mobile-bar
    assert "padding: 16px 16px 80px" in css
