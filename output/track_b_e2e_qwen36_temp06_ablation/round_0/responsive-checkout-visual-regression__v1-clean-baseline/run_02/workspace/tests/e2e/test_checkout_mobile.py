from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_compact_summary_no_negative_margin():
    """Compact summary must not use negative margin that causes overlap."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The mobile compact-summary block must NOT contain a negative margin-bottom
    # (which previously caused the summary to overlap the payment form).
    assert "margin-bottom: -64px" not in css


def test_mobile_shell_has_bottom_padding_for_fixed_bar():
    """Mobile checkout shell must have bottom padding to clear the fixed bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Inside the mobile media query, the shell needs padding-bottom so the
    # primary CTA is not hidden behind the fixed mobile bar.
    assert "padding: 16px 16px 80px" in css


def test_mobile_bar_is_fixed_and_visible():
    """The mobile bar must be fixed at the bottom on mobile viewports."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "position: fixed" in css
    assert "bottom: 0" in css
