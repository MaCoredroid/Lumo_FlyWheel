from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_mobile_compact_summary_does_not_overlap_payment():
    """Mobile viewport: compact summary must not use negative margin that overlaps payment content."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The negative margin-bottom that caused the overlap must be gone
    assert "margin-bottom: -64px" not in css

    # The mobile summary should have a positive margin-bottom for spacing
    assert "margin-bottom: 16px" in css


def test_mobile_shell_has_bottom_padding_for_fixed_bar():
    """Mobile viewport: checkout shell must have bottom padding to clear the fixed mobile bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The shell padding-bottom must be sufficient to clear the ~64px fixed bar
    assert "padding: 16px 16px 80px" in css


def test_mobile_bar_has_higher_z_index_than_sticky_summary():
    """Mobile viewport: fixed bottom bar z-index must exceed sticky summary z-index."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Summary z-index is 2, mobile bar must be 3 or higher
    assert "z-index: 3" in css


def test_mobile_bar_displays_as_flex():
    """Mobile viewport: checkout-mobile-bar must be visible as flex in compact mode."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert ".checkout-mobile-bar" in css
    # Within the mobile media query the bar should be display: flex
    media_section = css.split("@media (max-width: 600px)")[-1]
    assert "display: flex" in media_section
